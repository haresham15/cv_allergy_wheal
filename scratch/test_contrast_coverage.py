import os, sys, cv2, numpy as np
from skimage.feature import blob_log

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

try:
    from backend.services.preprocessing import preprocess
except ImportError:
    from services.preprocessing import preprocess

img = cv2.imread("Testphotos/allergy-Testing.jpg")
prep = preprocess(img)
l_clahe = prep["l_clahe"]
H, W = l_clahe.shape[:2]

# 1. Skin mask
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 130, 70]), np.array([255, 180, 135]))
mask_hsv = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([50, 255, 255]))
skin_raw = cv2.bitwise_and(mask_ycrcb, mask_hsv)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
skin_mask = cv2.morphologyEx(skin_raw, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
skin_clean = np.zeros((H, W), dtype=np.uint8)
if contours:
    max_area = max(cv2.contourArea(c) for c in contours)
    for c in contours:
        if cv2.contourArea(c) >= max(5000, 0.05 * max_area):
            cv2.drawContours(skin_clean, [c], -1, 255, -1)

# 2. Blobs
blobs = blob_log(l_clahe, min_sigma=2, max_sigma=25, num_sigma=10, threshold=0.040)
print(f"Total raw blobs: {len(blobs)}")

# Compute local contrast for each blob
scored_blobs = []
for b in blobs:
    r, c, s = int(round(b[0])), int(round(b[1])), int(round(b[2]))
    if r < 12 or r > H - 12 or c < 12 or c > W - 12:
        continue
    # Must be on skin
    if skin_clean[r, c] == 0:
        continue
    r0, r1 = max(0, r - 2), min(H, r + 3)
    c0, c1 = max(0, c - 2), min(W, c + 3)
    center_val = float(l_clahe[r0:r1, c0:c1].mean())
    R_outer = int(round(s * 2))
    r_out0, r_out1 = max(0, r - R_outer), min(H, r + R_outer + 1)
    c_out0, c_out1 = max(0, c - R_outer), min(W, c + R_outer + 1)
    outer_val = float(l_clahe[r_out0:r_out1, c_out0:c_out1].mean())
    contrast = abs(center_val - outer_val)
    scored_blobs.append((contrast, (float(c), float(r))))

# Sort by contrast descending
scored_blobs.sort(key=lambda x: -x[0])

# Spatial deduplication
candidates = []
for contrast, (cx, cy) in scored_blobs:
    if not any(np.hypot(cx - c[0], cy - c[1]) < 9 for c in candidates):
        candidates.append((cx, cy))

print(f"Total skin-filtered, contrast-ranked, deduplicated candidates: {len(candidates)}")

# Check against all 49 GT wheals
gt_mask = cv2.imread("Testphotos/allergy-Testing_mask.png", cv2.IMREAD_GRAYSCALE)
gt_bin = (cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
gt_contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
gt_wheals = [cv2.minEnclosingCircle(c) for c in gt_contours if cv2.contourArea(c) >= 10]

matched_gt = 0
for (gcx, gcy), gr in gt_wheals:
    min_d = min(np.hypot(gcx - cx, gcy - cy) for cx, cy in candidates)
    if min_d <= max(18.0, gr):
        matched_gt += 1
    else:
        print(f"  Missed GT at ({gcx:.1f}, {gcy:.1f}), radius={gr:.1f}, min_d={min_d:.1f}")

print(f"Ground truth wheals covered by candidates: {matched_gt} / {len(gt_wheals)} ({matched_gt/len(gt_wheals)*100:.1f}%)")
