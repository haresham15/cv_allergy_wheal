import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("backend"))

def extract_skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Extract a robust skin mask using combined YCrCb and HSV color spaces.
    
    Covers diverse Fitzpatrick skin tones (I-VI) while rejecting background,
    clothing, walls, and posters.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    
    # YCrCb skin rule: Cr in [133, 173], Cb in [77, 127]
    # Relaxed slightly to include erythema/wheals and darker skin tones
    lower_ycrcb = np.array([0, 130, 70], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # HSV skin rule: Hue in [0, 50] (reds, peaches, browns)
    lower_hsv = np.array([0, 20, 40], dtype=np.uint8)
    upper_hsv = np.array([50, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Combine masks
    combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    
    # Morphological closing to bridge small gaps and wheal edemas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    
    # Find contours and keep components that constitute meaningful skin area
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback if lighting is extreme: return full frame
        return np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255
    
    # Keep contours that have area > 5% of largest contour or > 5000px
    max_area = max(cv2.contourArea(c) for c in contours)
    skin_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for c in contours:
        if cv2.contourArea(c) >= max(5000, 0.05 * max_area):
            cv2.drawContours(skin_mask, [c], -1, 255, -1) # Fill holes completely
            
    return skin_mask

# Test on allergy-Testing.jpg
img = cv2.imread("Testphotos/allergy-Testing.jpg")
skin_m = extract_skin_mask(img)
h, w = img.shape[:2]
skin_coverage = (skin_m > 0).sum() / (h * w) * 100
print(f"allergy-Testing.jpg skin coverage: {skin_coverage:.1f}%")

# Test on the ground truth mask to ensure all 49 wheals are covered by the skin mask!
gt_mask = cv2.imread("Testphotos/allergy-Testing_mask.png", cv2.IMREAD_GRAYSCALE)
gt_bin = (gt_mask > 127).astype(np.uint8)
missed_by_skin = np.logical_and(gt_bin > 0, skin_m == 0).sum()
total_gt_px = (gt_bin > 0).sum()
print(f"Ground truth wheal pixels outside skin mask: {missed_by_skin} / {total_gt_px} ({missed_by_skin/total_gt_px*100:.2f}%)")
