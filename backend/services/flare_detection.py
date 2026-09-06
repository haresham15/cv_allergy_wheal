import cv2
import numpy as np
from sklearn.cluster import KMeans

def detect_flare(image_bgr: np.ndarray, wheal_contour: np.ndarray, ppm: float) -> float:
    """
    Uses traditional Machine Learning (K-Means Clustering) to detect the erythema (red flare)
    surrounding a detected wheal.

    Parameters
    ----------
    image_bgr : np.ndarray
        The original BGR image.
    wheal_contour : np.ndarray
        The contour of the detected wheal.
    ppm : float
        Pixels per millimeter for scale calculation.

    Returns
    -------
    float
        The estimated area of the flare in mm^2.
    """
    # 1. Create a bounding box around the wheal, expanded to capture the flare
    x, y, w, h = cv2.boundingRect(wheal_contour)
    
    # Expand bounding box by roughly 10mm (converted to pixels) in each direction
    padding = int(10 * ppm)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_bgr.shape[1], x + w + padding)
    y2 = min(image_bgr.shape[0], y + h + padding)
    
    roi_bgr = image_bgr[y1:y2, x1:x2]
    
    # If the ROI is too small, return 0
    if roi_bgr.shape[0] < 5 or roi_bgr.shape[1] < 5:
        return 0.0

    # 2. Convert ROI to LAB color space for better color discrimination (A channel = Green to Red)
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    
    # Reshape for K-Means (pixels x 3 channels)
    pixels = roi_lab.reshape((-1, 3))
    
    # 3. K-Means Clustering (k=3: Normal Skin, Red Flare, Wheal/Highlight)
    # We use scikit-learn for robustness and to demonstrate traditional ML skills.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=5)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    
    # 4. Identify the "Reddest" cluster based on the 'A' channel in LAB (index 1)
    # High 'A' value means more red.
    red_cluster_idx = np.argmax(centers[:, 1])
    
    # 5. Create a mask of just the red cluster
    flare_mask_1d = (labels == red_cluster_idx).astype(np.uint8)
    flare_mask = flare_mask_1d.reshape(roi_lab.shape[:2])
    
    # 6. Exclude the original wheal itself from the flare area
    # Shift wheal contour to ROI coordinates
    shifted_contour = wheal_contour.copy()
    shifted_contour[:, 0, 0] -= x1
    shifted_contour[:, 0, 1] -= y1
    
    wheal_mask = np.zeros(flare_mask.shape, dtype=np.uint8)
    cv2.drawContours(wheal_mask, [shifted_contour], -1, 1, -1)
    
    # The true flare is the red cluster MINUS the actual wheal body
    true_flare_mask = cv2.bitwise_and(flare_mask, flare_mask, mask=(1 - wheal_mask))
    
    # Optional: Apply morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    true_flare_mask = cv2.morphologyEx(true_flare_mask, cv2.MORPH_OPEN, kernel)
    
    # 7. Calculate Area
    flare_px_area = np.sum(true_flare_mask)
    flare_mm2 = flare_px_area / (ppm ** 2)
    
    return round(float(flare_mm2), 2)
