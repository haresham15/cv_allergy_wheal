import cv2
import numpy as np

from ..core import config


def find_wheals(image: np.ndarray, ppm: float):
    """Find wheal-like contours and return measurements + binary mask.

    Returns: (wheals_list, binary_mask)
    - wheals_list: list of dicts with keys: ellipse, center_point, diameter_mm, confidence
    - binary_mask: binary image showing detected wheal regions
    """
    # Convert to LAB color space for better skin/wheal separation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)

    # Create a mask for darker areas (wheals are typically darker than surrounding skin)
    # Use L channel threshold
    _, binary = cv2.threshold(l_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)  # Invert so wheals are white

    # Morphological operations to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    min_area_pixels = config.MIN_WHEAL_AREA_MM2 * (ppm ** 2)
    max_area_pixels = 200 * (ppm ** 2)  # Max ~200 mm^2
    cid = 1
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter by size
        if area < min_area_pixels or area > max_area_pixels:
            continue
        if len(cnt) < 5:
            continue

        # Fit ellipse
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy) = ellipse[0]
        (major, minor) = ellipse[1]
        major = max(major, minor)
        diameter_mm = major / ppm

        # Confidence: ratio of contour area to ellipse area
        ellipse_area = np.pi * (major / 2.0) * (minor / 2.0)
        confidence = min(1.0, float(area) / (ellipse_area + 1e-6))

        # Filter based on circularity (wheals should be fairly round)
        circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-6)
        if circularity < 0.5:
            continue

        results.append(
            {
                "id": cid,
                "ellipse": ellipse,
                "center_point": (float(cx), float(cy)),
                "diameter_mm": float(diameter_mm),
                "confidence": float(confidence),
            }
        )
        cid += 1

    return results, binary


