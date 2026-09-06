import os, sys, cv2, numpy as np

sys.path.insert(0, os.path.abspath("backend"))
from services.preprocessing import preprocess
from services.calibration import get_calibration

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
        })

print(f"Total GT wheals: {len(gt_wheals)}")

# In evaluate_on_testphotos.py, the matching condition is:
# dist <= max(max_dist_px, g["radius"])
# Where max_dist_px = 18.0!
# Notice: For a large wheal where radius is e.g. 25px (diameter 10mm), max(18.0, 25.0) = 25.0px!
# In our test_refinements.py, we strictly checked best_dist < 18.0 without taking into account max(18.0, gt["radius"])!

print("\nChecking matching with max(18.0, gt['radius']):")
test_detections = [
    {"center": (456.0, 70.0), "D": 7.89, "id": "FP#17"},
    {"center": (77.0, 137.0), "D": 3.25, "id": "FP#1"},
    {"center": (328.0, 178.0), "D": 2.01, "id": "FP#2"},
    {"center": (86.0, 91.0), "D": 2.25, "id": "FP#3"},
    {"center": (249.0, 141.0), "D": 2.78, "id": "FP#4"},
    {"center": (446.0, 161.0), "D": 3.05, "id": "FP#8"},
    {"center": (310.0, 214.0), "D": 2.50, "id": "FP#10"},
]

for td in test_detections:
    best_dist = 999
    best_gt = None
    for gt in gt_wheals:
        dist = np.hypot(td["center"][0] - gt["center"][0], td["center"][1] - gt["center"][1])
        if dist < best_dist:
            best_dist = dist
            best_gt = gt
    thresh = max(18.0, best_gt["radius"])
    print(f"{td['id']} center={td['center']} -> nearest GT#{best_gt['id']} at center={best_gt['center']}, dist={best_dist:.2f}px, allowed_thresh={thresh:.2f}px => {'MATCH!' if best_dist <= thresh else 'NO'}")
