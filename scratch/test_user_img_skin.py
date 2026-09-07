import os, sys, cv2, numpy as np

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

# Test on test photo
img = cv2.imread("Testphotos/allergy-Testing.jpg")
m = extract_skin_mask(img)
print("allergy-Testing.jpg skin coverage:", (m > 0).mean() * 100)

# Check user artifact if available
user_p = r"C:\Users\hares\.gemini\antigravity-ide\brain\df223ea3-7e14-474b-a389-94190dd895d3\.user_uploaded\media_1788725808313.png"
if os.path.exists(user_p):
    uimg = cv2.imread(user_p)
    um = extract_skin_mask(uimg)
    print("User uploaded image skin coverage:", (um > 0).mean() * 100)
