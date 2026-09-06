import os
import sys
import cv2
import numpy as np
import json

sys.path.insert(0, os.path.abspath("backend"))

from services.preprocessing import preprocess
from services.calibration import get_calibration
from services.segmentation import find_wheals

img_path = "Testphotos/allergy-Testing.jpg"
mask_path = "Testphotos/allergy-Testing_mask.png"

image = cv2.imread(img_path)
prep = preprocess(image)
cal = get_calibration(prep["resized"])
ppm = cal.ppm
H, W = prep["resized"].shape[:2]

gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if gt_mask.shape[:2] != (H, W):
    gt_mask = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)
gt_bin = (gt_mask > 127).astype(np.uint8)

gt_contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
gt_wheals = []
for idx, c in enumerate(gt_contours):
    area = cv2.contourArea(c)
    if area >= 10:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        d_circle = (r * 2) / ppm
        d_equiv = 2.0 * np.sqrt(area / np.pi) / ppm
        gt_wheals.append({
            "id": idx,
            "center": (float(cx), float(cy)),
            "radius": float(r),
            "area_px": float(area),
            "d_circle": float(d_circle),
            "d_equiv": float(d_equiv),
            "contour": c
        })

print(f"Total Ground Truth Wheals: {len(gt_wheals)}")

detections = find_wheals(prep, ppm=ppm)
print(f"Total Detections from SAM: {len(detections)}")

# Match detections to GT using IoU and Distance
gt_matched = set()
tp_list = []
fp_list = []

for det in detections:
    best_gt = None
    best_dist = float("inf")
    for gt in gt_wheals:
        dist = np.hypot(det.center[0] - gt["center"][0], det.center[1] - gt["center"][1])
        if dist < best_dist:
            best_dist = dist
            best_gt = gt

    # Match threshold: within 18 px (~3.3mm)
    if best_dist < 18.0 and best_gt["id"] not in gt_matched:
        gt_matched.add(best_gt["id"])
        tp_list.append({
            "det": det,
            "gt": best_gt,
            "dist": best_dist,
            "det_d": det.diameter_mm,
            "gt_circle_d": best_gt["d_circle"],
            "gt_equiv_d": best_gt["d_equiv"],
            "conf": det.confidence
        })
    else:
        fp_list.append({
            "det": det,
            "min_dist_to_any_gt": best_dist,
            "conf": det.confidence,
            "area_px": det.area_px,
            "diam_mm": det.diameter_mm
        })

fn_list = [gt for gt in gt_wheals if gt["id"] not in gt_matched]

print(f"\n--- MATCHING ANALYSIS ---")
print(f"True Positives:  {len(tp_list)}")
print(f"False Positives: {len(fp_list)}")
print(f"False Negatives: {len(fn_list)}")
print(f"Recall:    {len(tp_list)/len(gt_wheals)*100:.1f}%")
print(f"Precision: {len(tp_list)/len(detections)*100:.1f}%")
print(f"F1 Score:  {2*len(tp_list)/(len(gt_wheals)+len(detections))*100:.1f}%")

# Analyze Diameter Measurement Accuracy
circle_maes = [abs(tp["det_d"] - tp["gt_circle_d"]) for tp in tp_list]
equiv_maes = [abs(tp["det_d"] - tp["gt_equiv_d"]) for tp in tp_list]

print(f"\n--- DIAMETER MEASUREMENT MAE ---")
print(f"MAE vs Enclosing Circle: {np.mean(circle_maes):.2f} mm")
print(f"MAE vs Equivalent Area Diameter: {np.mean(equiv_maes):.2f} mm")

# Analyze False Positives
print(f"\n--- FALSE POSITIVES BREAKDOWN ({len(fp_list)}) ---")
fp_confs = [fp["conf"] for fp in fp_list]
fp_areas = [fp["area_px"] for fp in fp_list]
fp_diams = [fp["diam_mm"] for fp in fp_list]
print(f"FP Confidence: min={min(fp_confs):.3f}, mean={np.mean(fp_confs):.3f}, max={max(fp_confs):.3f}")
print(f"FP Diameter (mm): min={min(fp_diams):.2f}, mean={np.mean(fp_diams):.2f}, max={max(fp_diams):.2f}")
print(f"FP Area (px): min={min(fp_areas):.1f}, mean={np.mean(fp_areas):.1f}, max={max(fp_areas):.1f}")

# Analyze False Negatives
print(f"\n--- FALSE NEGATIVES BREAKDOWN ({len(fn_list)}) ---")
for fn in fn_list:
    print(f"  Missed Wheal at center=({fn['center'][0]:.1f}, {fn['center'][1]:.1f}), diam={fn['d_circle']:.2f}mm, area={fn['area_px']:.1f}px")
