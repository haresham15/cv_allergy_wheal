import os, sys, cv2, numpy as np, torch

sys.path.insert(0, os.path.abspath("backend"))
from services.preprocessing import preprocess
from services.calibration import get_calibration
from services.segmentation import _load_sam, _is_wheal_shaped
from core import config

img_path = "Testphotos/allergy-Testing.jpg"
image = cv2.imread(img_path)
prep = preprocess(image)
cal = get_calibration(prep["resized"])
ppm = cal.ppm
H, W = prep["resized"].shape[:2]

gt_mask_raw = cv2.imread("Testphotos/allergy-Testing_mask.png", cv2.IMREAD_GRAYSCALE)
gt_mask = cv2.resize(gt_mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)
gt_mask_bin = (gt_mask > 127).astype(np.uint8)
gt_contours, _ = cv2.findContours(gt_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

gt_wheals = []
for c in gt_contours:
    area = cv2.contourArea(c)
    if area >= 10:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        diam_mm = (r * 2) / ppm
        gt_wheals.append({
            "center": (cx, cy),
            "radius": r,
            "diameter_mm": diam_mm,
            "area_px": area,
            "contour": c,
        })

predictor = _load_sam()
image_rgb = cv2.cvtColor(prep["sam_ready_image"], cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

from skimage.feature import blob_log
blobs = blob_log(prep["l_clahe"], min_sigma=2, max_sigma=25, num_sigma=10, threshold=0.045)
candidates = []
border_margin = 12
for b in sorted(blobs, key=lambda x: -x[2]):
    cx, cy = float(b[1]), float(b[0])
    if cx < border_margin or cx > (W - border_margin) or cy < border_margin or cy > (H - border_margin):
        continue
    if not any(np.hypot(cx - c[0], cy - c[1]) < 11 for c in candidates):
        candidates.append((cx, cy))

device = "cuda" if torch.cuda.is_available() else "cpu"
coords_np = np.array(candidates)[:, np.newaxis, :]
labels_np = np.ones((len(candidates), 1), dtype=np.int32)
coords_tf = predictor.transform.apply_coords(coords_np, (H, W))
coords_torch = torch.as_tensor(coords_tf, dtype=torch.float, device=device)
labels_torch = torch.as_tensor(labels_np, dtype=torch.int, device=device)

with torch.no_grad():
    masks_tensor, scores_tensor, _ = predictor.predict_torch(
        point_coords=coords_torch,
        point_labels=labels_torch,
        multimask_output=True,
    )

def evaluate_params(min_area_thresh, nms_dist_cap, conf_thresh):
    min_area_px = max(min_area_thresh, config.MIN_WHEAL_AREA_MM2 * (ppm ** 2))
    max_area_px = config.MAX_WHEAL_AREA_MM2 * (ppm ** 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw_results = []

    for i, center_pt in enumerate(candidates):
        best_cand = None
        best_score = -1.0
        for j in range(3):
            score = float(scores_tensor[i, j])
            if score < conf_thresh:
                continue
            mask_binary = masks_tensor[i, j].cpu().numpy().astype(np.uint8)
            area_px = float(np.sum(mask_binary))
            if area_px < min_area_px or area_px > max_area_px or area_px > 3000.0:
                continue
            mask_smooth = cv2.morphologyEx(mask_binary * 255, cv2.MORPH_CLOSE, kernel)
            mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            c_area = float(cv2.contourArea(contour))
            if c_area < min_area_px:
                continue
            if not _is_wheal_shaped(contour, c_area):
                continue
            (_, _), radius = cv2.minEnclosingCircle(contour)
            diameter_px = radius * 2.0
            diameter_mm = diameter_px / ppm
            if diameter_mm < 1.0 or diameter_mm > 40.0:
                continue
            if score > best_score:
                best_score = score
                best_cand = {
                    "mask": mask_binary,
                    "contour": contour,
                    "center": center_pt,
                    "radius": radius,
                    "diameter_mm": diameter_mm,
                    "area_px": c_area,
                    "confidence": score,
                }
        if best_cand is not None:
            raw_results.append(best_cand)

    raw_results.sort(key=lambda item: -item["confidence"])
    deduped = []
    for cand in raw_results:
        is_dup = False
        for kept in deduped:
            intersection = np.logical_and(cand["mask"], kept["mask"]).sum()
            union = np.logical_or(cand["mask"], kept["mask"]).sum()
            if union > 0 and (intersection / union) > 0.28:
                is_dup = True
                break
            dist = np.hypot(cand["center"][0] - kept["center"][0], cand["center"][1] - kept["center"][1])
            # Adaptive distance suppression
            if dist < min(cand["radius"], kept["radius"]) * 0.80 and dist < nms_dist_cap:
                is_dup = True
                break
        if not is_dup:
            deduped.append(cand)

    # Global matching
    pairs = []
    for gi, g in enumerate(gt_wheals):
        for pi, p in enumerate(deduped):
            dist = np.hypot(g["center"][0] - p["center"][0], g["center"][1] - p["center"][1])
            if dist <= max(18.0, g["radius"]):
                pairs.append((dist, gi, pi))
    pairs.sort(key=lambda x: x[0])
    matched_gt = set()
    matched_pred = set()
    matches = []
    for dist, gi, pi in pairs:
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
            matches.append(abs(gt_wheals[gi]["diameter_mm"] - deduped[pi]["diameter_mm"]))

    tp = len(matched_gt)
    fp = len(deduped) - tp
    rec = tp / len(gt_wheals) * 100.0
    prec = tp / len(deduped) * 100.0 if len(deduped) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    mae = np.mean(matches) if matches else 0.0

    # Composite mask IoU
    comp = np.zeros((H, W), dtype=np.uint8)
    for d in deduped:
        cv2.drawContours(comp, [d["contour"]], -1, 1, -1)
    inter = np.logical_and(comp, gt_mask_bin).sum()
    union = np.logical_or(comp, gt_mask_bin).sum()
    iou = inter / union if union > 0 else 0

    return tp, fp, len(deduped), rec, prec, f1, mae, iou

print("\n--- Sweeping min_area, nms_dist_cap, conf_thresh ---")
for area_th in [20, 35, 45]:
    for dist_cap in [18.0, 25.0, 30.0]:
        for conf in [0.70, 0.72]:
            tp, fp, tot, rec, prec, f1, mae, iou = evaluate_params(area_th, dist_cap, conf)
            print(f"Area={area_th:2d}, DistCap={dist_cap:4.1f}, Conf={conf:.2f} | Det={tot:2d}, TP={tp:2d}, FP={fp:2d} | Rec={rec:.1f}%, Prec={prec:.1f}%, F1={f1:.1f}% | MAE={mae:.2f}mm, IoU={iou:.4f}")
