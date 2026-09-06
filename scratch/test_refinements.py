import os
import sys
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("backend"))
from core import config
from services.preprocessing import preprocess
from services.calibration import get_calibration
from services.segmentation import _load_sam, _is_wheal_shaped, _classify_severity, WhealResult

img_path = "Testphotos/allergy-Testing.jpg"
mask_path = "Testphotos/allergy-Testing_mask.png"

image = cv2.imread(img_path)
prep = preprocess(image)
cal = get_calibration(prep["resized"])
ppm = cal.ppm
H, W = prep["resized"].shape[:2]

gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
gt_bin = (cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
gt_contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
gt_wheals = []
for idx, c in enumerate(gt_contours):
    area = cv2.contourArea(c)
    if area >= 10:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        # Clinical standard: equivalent circular diameter or ellipse axis mean
        d_equiv = 2.0 * np.sqrt(area / np.pi) / ppm
        gt_wheals.append({
            "id": idx,
            "center": (float(cx), float(cy)),
            "radius": float(r),
            "area_px": float(area),
            "diam_mm": float(d_equiv),
            "contour": c
        })

print(f"Ground truth wheals: {len(gt_wheals)}")

def run_refined_segmentation(
    conf_thresh=0.70,
    border_margin=15,
    nms_iou=0.30,
    nms_dist_ratio=0.70,
    use_equiv_diam=True,
    color_contrast_check=True,
):
    predictor = _load_sam()
    image_rgb = cv2.cvtColor(prep["sam_ready_image"], cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(prep["sam_ready_image"], cv2.COLOR_BGR2LAB)
    predictor.set_image(image_rgb)

    # 1. Multi-scale LoG with border margin filtering
    from skimage.feature import blob_log
    blobs = blob_log(prep["l_clahe"], min_sigma=2, max_sigma=25, num_sigma=10, threshold=0.040)

    candidates = []
    for b in sorted(blobs, key=lambda x: -x[2]):
        cx, cy = float(b[1]), float(b[0])
        # Border margin filter: exclude edge clipping artifacts
        if cx < border_margin or cx > (W - border_margin) or cy < border_margin or cy > (H - border_margin):
            continue
        # Spatial deduplication
        if not any(np.hypot(cx - c[0], cy - c[1]) < 10 for c in candidates):
            candidates.append((cx, cy))

    if len(candidates) > 200:
        candidates = candidates[:200]

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

    min_area_px = max(config.SAM_MIN_MASK_REGION_AREA, config.MIN_WHEAL_AREA_MM2 * (ppm ** 2))
    max_area_px = config.MAX_WHEAL_AREA_MM2 * (ppm ** 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # gentler 3x3 smoothing
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

            # Discard diffuse/huge background masks (>3,000 px) or tiny noise (<30 px)
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

            # Clinical diameter: Equivalent circular diameter or min enclosing circle
            if use_equiv_diam:
                diameter_px = 2.0 * np.sqrt(c_area / np.pi)
            else:
                (_, _), r = cv2.minEnclosingCircle(contour)
                diameter_px = r * 2.0
            diameter_mm = diameter_px / ppm

            if diameter_mm < 1.0 or diameter_mm > 35.0:
                continue

            # Optional color contrast check: wheal center vs ring background
            if color_contrast_check:
                mask_cnt = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(mask_cnt, [contour], -1, 1, -1)
                dilated = cv2.dilate(mask_cnt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
                ring = dilated - mask_cnt
                
                # Check lightness and erythema (a* channel in LAB)
                if ring.sum() > 10 and mask_cnt.sum() > 10:
                    a_inside = image_lab[:, :, 1][mask_cnt > 0].mean()
                    a_ring = image_lab[:, :, 1][ring > 0].mean()
                    l_inside = image_lab[:, :, 0][mask_cnt > 0].mean()
                    l_ring = image_lab[:, :, 0][ring > 0].mean()
                    # A true wheal has higher erythema (redness a*) or slight edema pallor/flare
                    # Discard uniform background skin where difference is zero
                    delta = abs(a_inside - a_ring) + abs(l_inside - l_ring)
                    if delta < 0.8:
                        continue

            if score > best_score:
                best_score = score
                best_cand = {
                    "mask": mask_binary,
                    "contour": contour,
                    "center": center_pt,
                    "diameter_mm": diameter_mm,
                    "area_px": c_area,
                    "confidence": score,
                }

        if best_cand is not None:
            raw_results.append(best_cand)

    # 3. Enhanced NMS
    raw_results.sort(key=lambda item: -item["confidence"])
    deduped = []
    for cand in raw_results:
        is_dup = False
        for kept in deduped:
            intersection = np.logical_and(cand["mask"], kept["mask"]).sum()
            union = np.logical_or(cand["mask"], kept["mask"]).sum()
            if union > 0 and (intersection / union) > nms_iou:
                is_dup = True
                break
            dist = np.hypot(cand["center"][0] - kept["center"][0], cand["center"][1] - kept["center"][1])
            min_r = min(cand["diameter_mm"], kept["diameter_mm"]) * ppm / 2.0
            if dist < min_r * nms_dist_ratio and dist < 14.0:
                is_dup = True
                break
        if not is_dup:
            deduped.append(cand)

    return deduped

print("\n--- Testing Baseline vs Refined Segmentation ---")
for conf in [0.68, 0.70, 0.72]:
    for border in [12, 15]:
        results = run_refined_segmentation(conf_thresh=conf, border_margin=border)
        
        # Match against ground truth
        matched_gt = set()
        tp_list = []
        for det in results:
            best_gt = None
            best_dist = float("inf")
            for gt in gt_wheals:
                dist = np.hypot(det["center"][0] - gt["center"][0], det["center"][1] - gt["center"][1])
                if dist < best_dist:
                    best_dist = dist
                    best_gt = gt
            if best_dist < 18.0 and best_gt["id"] not in matched_gt:
                matched_gt.add(best_gt["id"])
                tp_list.append((det, best_gt))

        tp = len(tp_list)
        fp = len(results) - tp
        rec = tp / len(gt_wheals) * 100
        prec = tp / len(results) * 100 if len(results) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        # Diameter MAE on True Positives
        maes = [abs(t[0]["diameter_mm"] - t[1]["diam_mm"]) for t in tp_list]
        mean_mae = np.mean(maes) if maes else 0.0

        # Mask IoU
        pred_mask_full = np.zeros((H, W), dtype=np.uint8)
        for r in results:
            cv2.drawContours(pred_mask_full, [r["contour"]], -1, 1, -1)
        inter = np.logical_and(pred_mask_full, gt_bin).sum()
        union = np.logical_or(pred_mask_full, gt_bin).sum()
        iou = inter / union if union > 0 else 0

        print(f"Conf={conf:.2f}, Border={border}: Det={len(results)}, TP={tp}, FP={fp}, Rec={rec:.1f}%, Prec={prec:.1f}%, F1={f1:.1f}%, MAE={mean_mae:.2f}mm, IoU={iou:.4f}")
