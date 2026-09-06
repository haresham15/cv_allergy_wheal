import os
import sys
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("backend"))
from core import config
from services.preprocessing import preprocess
from services.calibration import get_calibration
from services.segmentation import _load_sam, _is_wheal_shaped

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
        d_equiv = 2.0 * np.sqrt(area / np.pi) / ppm
        gt_wheals.append({
            "id": idx,
            "center": (float(cx), float(cy)),
            "radius": float(r),
            "area_px": float(area),
            "diam_mm": float(d_equiv),
            "contour": c
        })

predictor = _load_sam()
image_rgb = cv2.cvtColor(prep["sam_ready_image"], cv2.COLOR_BGR2RGB)
image_lab = cv2.cvtColor(prep["sam_ready_image"], cv2.COLOR_BGR2LAB)
predictor.set_image(image_rgb)

from skimage.feature import blob_log
blobs = blob_log(prep["l_clahe"], min_sigma=2, max_sigma=25, num_sigma=10, threshold=0.038)

candidates = []
border_margin = 12
for b in sorted(blobs, key=lambda x: -x[2]):
    cx, cy = float(b[1]), float(b[0])
    if cx < border_margin or cx > (W - border_margin) or cy < border_margin or cy > (H - border_margin):
        continue
    if not any(np.hypot(cx - c[0], cy - c[1]) < 10 for c in candidates):
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

min_area_px = max(config.SAM_MIN_MASK_REGION_AREA, config.MIN_WHEAL_AREA_MM2 * (ppm ** 2))
max_area_px = config.MAX_WHEAL_AREA_MM2 * (ppm ** 2)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
raw_results = []

for i, center_pt in enumerate(candidates):
    best_cand = None
    best_score = -1.0

    for j in range(3):
        score = float(scores_tensor[i, j])
        if score < 0.70:
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

        diameter_px = 2.0 * np.sqrt(c_area / np.pi)
        diameter_mm = diameter_px / ppm

        if diameter_mm < 1.0 or diameter_mm > 35.0:
            continue

        mask_cnt = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask_cnt, [contour], -1, 1, -1)
        dilated = cv2.dilate(mask_cnt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        ring = dilated - mask_cnt
        
        delta_color = 0.0
        if ring.sum() > 10 and mask_cnt.sum() > 10:
            a_inside = image_lab[:, :, 1][mask_cnt > 0].mean()
            a_ring = image_lab[:, :, 1][ring > 0].mean()
            l_inside = image_lab[:, :, 0][mask_cnt > 0].mean()
            l_ring = image_lab[:, :, 0][ring > 0].mean()
            delta_color = abs(a_inside - a_ring) + abs(l_inside - l_ring)

        if score > best_score:
            best_score = score
            best_cand = {
                "mask": mask_binary,
                "contour": contour,
                "center": center_pt,
                "diameter_mm": diameter_mm,
                "area_px": c_area,
                "confidence": score,
                "delta_color": delta_color
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
        if union > 0 and (intersection / union) > 0.30:
            is_dup = True
            break
        dist = np.hypot(cand["center"][0] - kept["center"][0], cand["center"][1] - kept["center"][1])
        min_r = min(cand["diameter_mm"], kept["diameter_mm"]) * ppm / 2.0
        if dist < min_r * 0.70 and dist < 14.0:
            is_dup = True
            break
    if not is_dup:
        deduped.append(cand)

matched_gt = set()
tp_list = []
fp_list = []
for det in deduped:
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
    else:
        fp_list.append(det)

fn_list = [gt for gt in gt_wheals if gt["id"] not in matched_gt]

print(f"Total Detections: {len(deduped)}, TP: {len(tp_list)}, FP: {len(fp_list)}, FN: {len(fn_list)}")
print("\n--- Remaining False Positives ---")
for i, fp in enumerate(fp_list):
    print(f"FP #{i+1}: center=({fp['center'][0]:.1f}, {fp['center'][1]:.1f}), D={fp['diameter_mm']:.2f}mm, conf={fp['confidence']:.3f}, delta_color={fp['delta_color']:.2f}, area={fp['area_px']:.0f}px")

print("\n--- Remaining False Negatives (Missed GT) ---")
for i, fn in enumerate(fn_list):
    print(f"FN #{i+1}: id={fn['id']}, center=({fn['center'][0]:.1f}, {fn['center'][1]:.1f}), D={fn['diam_mm']:.2f}mm, area={fn['area_px']:.0f}px")
