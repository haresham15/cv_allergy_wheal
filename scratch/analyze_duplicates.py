import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath("backend"))
from services.preprocessing import preprocess
from services.calibration import get_calibration
from services.segmentation import find_wheals

image = cv2.imread("Testphotos/allergy-Testing.jpg")
prep = preprocess(image)
cal = get_calibration(prep["resized"])
ppm = cal.ppm
H, W = prep["resized"].shape[:2]

gt_mask = cv2.imread("Testphotos/allergy-Testing_mask.png", cv2.IMREAD_GRAYSCALE)
gt_bin = (cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
gt_contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
gt_wheals = []
for idx, c in enumerate(gt_contours):
    area = cv2.contourArea(c)
    if area >= 10:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        gt_wheals.append({"id": idx, "center": (float(cx), float(cy)), "r": float(r), "area": float(area)})

detections = find_wheals(prep, ppm=ppm)

matched_gt = {}
unmatched_detections = []

for d in detections:
    dists = [(idx, np.hypot(d.center[0] - g["center"][0], d.center[1] - g["center"][1])) for idx, g in enumerate(gt_wheals)]
    dists.sort(key=lambda x: x[1])
    closest_gt_idx, closest_dist = dists[0]

    if closest_dist < 18.0:
        if closest_gt_idx not in matched_gt:
            matched_gt[closest_gt_idx] = [d]
        else:
            matched_gt[closest_gt_idx].append(d)
    else:
        unmatched_detections.append((d, closest_dist))

print(f"Total Detections: {len(detections)}")
print(f"GT wheals with exactly 1 detection: {sum(1 for gts in matched_gt.values() if len(gts) == 1)}")
double_count_gts = [g_id for g_id, gts in matched_gt.items() if len(gts) > 1]
print(f"GT wheals with >1 detections (Double-counts): {len(double_count_gts)}")
total_extra_detections = sum(len(gts) - 1 for gts in matched_gt.values())
print(f"Total duplicate detections on true wheals: {total_extra_detections}")

for g_id in double_count_gts:
    d_list = matched_gt[g_id]
    c = gt_wheals[g_id]["center"]
    print(f"\n  GT {g_id} at ({c[0]:.1f}, {c[1]:.1f}) has {len(d_list)} detections:")
    for d in d_list:
        print(f"    -> Det at ({d.center[0]:.1f}, {d.center[1]:.1f}), diam={d.diameter_mm:.1f}mm, conf={d.confidence:.2f}")

print(f"\nCompletely isolated False Positives (>18px from any GT): {len(unmatched_detections)}")
for d, dist in unmatched_detections:
    print(f"  Isolated FP at ({d.center[0]:.1f}, {d.center[1]:.1f}), dist_to_nearest_gt={dist:.1f}px, diam={d.diameter_mm:.1f}mm, conf={d.confidence:.2f}")
