import os
import sys
import time
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("backend"))
from services.preprocessing import preprocess
from services.calibration import get_calibration
from services.segmentation import _load_sam, _is_wheal_shaped, _classify_severity, WhealResult
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
    return {"matches": matches, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

def extract_skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 128, 70], dtype=np.uint8), np.array([255, 180, 135], dtype=np.uint8))
    mask_hsv = cv2.inRange(hsv, np.array([0, 18, 35], dtype=np.uint8), np.array([50, 255, 255], dtype=np.uint8))
    combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_large)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image_bgr.shape[:2]
    total_area = h * w
    skin_clean = np.zeros((h, w), dtype=np.uint8)
    if contours:
        max_area = max(cv2.contourArea(c) for c in contours)
        for c in contours:
            c_area = cv2.contourArea(c)
            if c_area >= max(3000, 0.05 * max_area):
                cv2.drawContours(skin_clean, [c], -1, 255, -1)
    if (skin_clean > 0).sum() < 0.05 * total_area:
        skin_clean = np.ones((h, w), dtype=np.uint8) * 255
    return skin_clean

# Load Benchmark
img_path = "Testphotos/allergy-Testing.jpg"
image = cv2.imread(img_path)
prep = preprocess(image)
skin_mask = extract_skin_mask(prep["resized"])
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
        gt_wheals.append({
            "center": (cx, cy),
            "radius": r,
            "diameter_mm": (r * 2) / ppm,
            "area_px": area,
            "contour": c,
        })

predictor = _load_sam()
image_rgb = cv2.cvtColor(prep["sam_ready_image"], cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

from skimage.feature import blob_log

blobs = blob_log(prep["l_clahe"], min_sigma=2, max_sigma=25, num_sigma=10, threshold=0.045)
border_margin = 12
scored_blobs = []
for b in blobs:
    cy, cx, sigma = float(b[0]), float(b[1]), float(b[2])
    r, c = int(round(cy)), int(round(cx))
    if r < border_margin or r > H - border_margin or c < border_margin or c > W - border_margin:
        continue
    # Skin check
    if skin_mask[r, c] == 0:
        continue
    # Response contrast
    r0, r1 = max(0, r - 2), min(H, r + 3)
    c0, c1 = max(0, c - 2), min(W, c + 3)
    center_val = float(prep["l_clahe"][r0:r1, c0:c1].mean())
    R_outer = int(round(sigma * 2))
    r_out0, r_out1 = max(0, r - R_outer), min(H, r + R_outer + 1)
    c_out0, c_out1 = max(0, c - R_outer), min(W, c + R_outer + 1)
    outer_val = float(prep["l_clahe"][r_out0:r_out1, c_out0:c_out1].mean())
    contrast = abs(center_val - outer_val)
    if contrast < 4.0:
        continue
    scored_blobs.append((contrast, (cx, cy)))

scored_blobs.sort(key=lambda item: -item[0])

candidates = []
for contrast, (cx, cy) in scored_blobs:
    if not any(np.hypot(cx - c[0], cy - c[1]) < 10 for c in candidates):
        candidates.append((cx, cy))

if len(candidates) > 280:
    candidates = candidates[:280]

print(f"Candidates generated: {len(candidates)}")

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

min_area_px = max(18.0, min(config.SAM_MIN_MASK_REGION_AREA, config.MIN_WHEAL_AREA_MM2 * (ppm ** 2)))
max_area_px = min(3500.0, config.MAX_WHEAL_AREA_MM2 * (ppm ** 2))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
raw_results = []

for i, center_pt in enumerate(candidates):
    best_candidate = None
    best_score = -1.0

    for j in range(3):
        score = float(scores_tensor[i, j])
        # Score floor: allow subtle wheals (>= 0.55) through
        if score < 0.55:
            continue

        mask_binary = masks_tensor[i, j].cpu().numpy().astype(np.uint8)
        area_px = float(np.sum(mask_binary))

        if area_px < min_area_px or area_px > max_area_px:
            continue

        # Skin overlap check: mask must be on skin
        if (mask_binary > 0).sum() > 0:
            outside_skin = np.logical_and(mask_binary, skin_mask == 0).sum() / area_px
            if outside_skin > 0.25:
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
        diameter_px = radius * 2
        diameter_mm = diameter_px / ppm

        if diameter_mm < 1.0 or diameter_mm > 40.0:
            continue

        if score > best_score:
            best_score = score
            area_mm2 = c_area / (ppm ** 2)
            severity = _classify_severity(diameter_mm)
            best_candidate = {
                "mask": mask_binary,
                "contour": contour,
                "center": center_pt,
                "radius": radius,
                "diameter_px": diameter_px,
                "diameter_mm": diameter_mm,
                "area_px": c_area,
                "area_mm2": area_mm2,
                "confidence": score,
                "severity": severity,
            }

    if best_candidate is not None:
        raw_results.append(best_candidate)

raw_results.sort(key=lambda item: -item["confidence"])
deduped_candidates = []
for cand in raw_results:
    is_dup = False
    for kept in deduped_candidates:
        intersection = np.logical_and(cand["mask"], kept["mask"]).sum()
        union = np.logical_or(cand["mask"], kept["mask"]).sum()
        if union > 0 and (intersection / union) > 0.30:
            is_dup = True
            break
        dist = np.hypot(cand["center"][0] - kept["center"][0],
                        cand["center"][1] - kept["center"][1])
        if dist < min(cand["radius"], kept["radius"]) * 0.75 and dist < 18.0:
            is_dup = True
            break

    if not is_dup:
        deduped_candidates.append(cand)

comp_mask = np.zeros((H, W), dtype=np.uint8)
for w in deduped_candidates:
    cv2.drawContours(comp_mask, [w["contour"]], -1, 1, -1)

match_res = match_detections(gt_wheals, deduped_candidates)
iou, dice = compute_mask_metrics(comp_mask, gt_mask_bin)
mae = np.mean([m["diam_error_mm"] for m in match_res["matches"]]) if match_res["matches"] else 999.0
mape = np.mean([m["diam_pct_error"] for m in match_res["matches"]]) if match_res["matches"] else 100.0

print(f"\nFINAL BENCHMARK RESULTS:")
print(f"  * Detected Wheals:     {len(deduped_candidates)} / {len(gt_wheals)}")
print(f"  * True Positives (TP): {match_res['tp']}")
print(f"  * False Positives (FP):{match_res['fp']}")
print(f"  * False Negatives (FN):{match_res['fn']}")
print(f"  * Detection Recall:    {match_res['recall']*100:.1f}%")
print(f"  * Detection Precision: {match_res['precision']*100:.1f}%")
print(f"  * Detection F1-Score:  {match_res['f1']*100:.1f}%")
print(f"  * Mean Diameter MAE:   {mae:.2f} mm")
print(f"  * Mean Diameter MAPE:  {mape:.1f}%")
print(f"  * Mask IoU:            {iou:.4f}")
print(f"  * Mask Dice:           {dice:.4f}")
