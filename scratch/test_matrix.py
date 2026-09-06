import os
import sys
import time
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("backend"))
from services.preprocessing import preprocess
from services.calibration import get_calibration
from services.segmentation import _load_sam, _is_wheal_shaped, WhealResult
from core import config

def compute_mask_metrics(pred_mask: np.ndarray, target_mask: np.ndarray):
    p = (pred_mask > 0).astype(np.uint8)
    t = (target_mask > 0).astype(np.uint8)
    intersection = int((p * t).sum())
    union = int(p.sum() + t.sum() - intersection)
    iou = 1.0 if union == 0 else float(intersection / union)
    dice = 1.0 if (p.sum() + t.sum()) == 0 else float(2 * intersection / (p.sum() + t.sum()))
    return iou, dice

def match_detections(gt_wheals: list, pred_wheals: list, max_dist_px: float = 18.0):
    matched_gt = set()
    matched_pred = set()
    matches = []
    pairs = []
    for gi, g in enumerate(gt_wheals):
        for pi, p in enumerate(pred_wheals):
            dist = np.hypot(g["center"][0] - p["center"][0], g["center"][1] - p["center"][1])
            if dist <= max(max_dist_px, g["radius"]):
                pairs.append((dist, gi, pi))

    pairs.sort(key=lambda x: x[0])

    for dist, gi, pi in pairs:
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
            matches.append({
                "gt_idx": gi,
                "pred_idx": pi,
                "dist": dist,
                "gt_diam_mm": gt_wheals[gi]["diameter_mm"],
                "pred_diam_mm": pred_wheals[pi]["diameter_mm"],
                "diam_error_mm": abs(gt_wheals[gi]["diameter_mm"] - pred_wheals[pi]["diameter_mm"]),
                "diam_pct_error": abs(gt_wheals[gi]["diameter_mm"] - pred_wheals[pi]["diameter_mm"]) / gt_wheals[gi]["diameter_mm"] * 100.0,
            })

    tp = len(matches)
    fp = len(pred_wheals) - tp
    fn = len(gt_wheals) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "matches": matches,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# 1. Load Data
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

print(f"Ground Truth contains {len(gt_wheals)} wheals.")

predictor = _load_sam()
image_rgb = cv2.cvtColor(prep["sam_ready_image"], cv2.COLOR_BGR2RGB)
image_lab = cv2.cvtColor(prep["sam_ready_image"], cv2.COLOR_BGR2LAB)
predictor.set_image(image_rgb)

from skimage.feature import blob_log

def test_pipeline(blob_thresh, conf_thresh, border_margin, kernel_size, diam_method, color_filter):
    blobs = blob_log(prep["l_clahe"], min_sigma=2, max_sigma=25, num_sigma=10, threshold=blob_thresh)
    candidates = []
    for b in sorted(blobs, key=lambda x: -x[2]):
        cx, cy = float(b[1]), float(b[0])
        if cx < border_margin or cx > (W - border_margin) or cy < border_margin or cy > (H - border_margin):
            continue
        if not any(np.hypot(cx - c[0], cy - c[1]) < 11 for c in candidates):
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    raw_results = []

    for i, center_pt in enumerate(candidates):
        best_candidate = None
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
            if diam_method == "enclosing":
                diameter_px = radius * 2.0
            elif diam_method == "equiv":
                diameter_px = 2.0 * np.sqrt(c_area / np.pi)
            elif diam_method == "ellipse" and len(contour) >= 5:
                _, (major, minor), _ = cv2.fitEllipse(contour)
                diameter_px = (major + minor) / 2.0
            else:
                diameter_px = radius * 2.0

            diameter_mm = diameter_px / ppm
            if diameter_mm < 0.5 or diameter_mm > 40.0:
                continue

            if color_filter:
                mask_cnt = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(mask_cnt, [contour], -1, 1, -1)
                dilated = cv2.dilate(mask_cnt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
                ring = dilated - mask_cnt
                if ring.sum() > 10 and mask_cnt.sum() > 10:
                    a_inside = image_lab[:, :, 1][mask_cnt > 0].mean()
                    a_ring = image_lab[:, :, 1][ring > 0].mean()
                    l_inside = image_lab[:, :, 0][mask_cnt > 0].mean()
                    l_ring = image_lab[:, :, 0][ring > 0].mean()
                    delta = abs(a_inside - a_ring) + abs(l_inside - l_ring)
                    if delta < 0.7:
                        continue

            if score > best_score:
                best_score = score
                best_candidate = {
                    "mask": mask_binary,
                    "contour": contour,
                    "center": center_pt,
                    "radius": radius,
                    "diameter_px": diameter_px,
                    "diameter_mm": diameter_mm,
                    "area_px": c_area,
                    "confidence": score,
                }

        if best_candidate is not None:
            raw_results.append(best_candidate)

    raw_results.sort(key=lambda item: -item["confidence"])
    deduped = []
    for cand in raw_results:
        is_dup = False
        for kept in deduped:
            intersection = np.logical_and(cand["mask"], kept["mask"]).sum()
            union = np.logical_or(cand["mask"], kept["mask"]).sum()
            if union > 0 and (intersection / union) > 0.30:
                is_dup = True
                break
            dist = np.hypot(cand["center"][0] - kept["center"][0], cand["center"][1] - kept["center"][1])
            if dist < min(cand["radius"], kept["radius"]) * 0.70 and dist < 15.0:
                is_dup = True
                break
        if not is_dup:
            deduped.append(cand)

    composite_mask = np.zeros((H, W), dtype=np.uint8)
    for w in deduped:
        cv2.drawContours(composite_mask, [w["contour"]], -1, 1, -1)

    match_res = match_detections(gt_wheals, deduped)
    iou, dice = compute_mask_metrics(composite_mask, gt_mask_bin)
    mae = np.mean([m["diam_error_mm"] for m in match_res["matches"]]) if match_res["matches"] else 999.0
    mape = np.mean([m["diam_pct_error"] for m in match_res["matches"]]) if match_res["matches"] else 100.0

    return {
        "detections": len(deduped),
        "tp": match_res["tp"],
        "fp": match_res["fp"],
        "fn": match_res["fn"],
        "recall": match_res["recall"] * 100.0,
        "precision": match_res["precision"] * 100.0,
        "f1": match_res["f1"] * 100.0,
        "mae": mae,
        "mape": mape,
        "iou": iou,
        "dice": dice
    }

print("\n--- Running Optimization Matrix ---")
experiments = [
    # blob_thresh, conf_thresh, border_margin, kernel_size, diam_method, color_filter
    (0.045, 0.60, 0, 5, "enclosing", False),  # Original baseline
    (0.045, 0.68, 12, 3, "enclosing", False),
    (0.045, 0.70, 12, 3, "enclosing", False),
    (0.045, 0.70, 12, 3, "equiv", False),
    (0.045, 0.70, 12, 3, "ellipse", False),
    (0.045, 0.70, 12, 3, "ellipse", True),
    (0.040, 0.70, 12, 3, "ellipse", True),
    (0.040, 0.68, 12, 3, "ellipse", True),
]

for exp in experiments:
    b_th, c_th, bm, ks, dm, cf = exp
    res = test_pipeline(b_th, c_th, bm, ks, dm, cf)
    name = f"b_th={b_th}, conf={c_th}, border={bm}, kern={ks}, diam={dm}, col={cf}"
    print(f"{name:58s} | Det={res['detections']:2d} TP={res['tp']:2d} FP={res['fp']:2d} | Rec={res['recall']:.1f}% Prec={res['precision']:.1f}% F1={res['f1']:.1f}% | MAE={res['mae']:.2f}mm MAPE={res['mape']:.1f}% | IoU={res['iou']:.4f}")
