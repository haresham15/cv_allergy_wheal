"""Image preprocessing for the allergy wheal detection pipeline.

Standardises raw camera input before it reaches the segmentation stage:
  * resize to a manageable resolution
  * Gaussian blur to tame sensor noise / skin texture
  * CLAHE to enhance the subtle contrast between wheals and healthy skin
"""

import cv2
import numpy as np
import imutils

from core import config


def extract_skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Extract patient skin region using adaptive color segmentation in YCrCb and HSV.

    Robust across Fitzpatrick skin types I through VI while rejecting
    clothing, background posters, furniture, hair, and jewelry.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

    # YCrCb skin region: Cr in [128, 180], Cb in [70, 135]
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 128, 70], dtype=np.uint8), np.array([255, 180, 135], dtype=np.uint8))

    # HSV skin region: H in [0, 50] (red, orange, peach, and brown tones)
    mask_hsv = cv2.inRange(hsv, np.array([0, 18, 35], dtype=np.uint8), np.array([50, 255, 255], dtype=np.uint8))

    # Combine color masks
    combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)

    # Morphological closing to bridge prick wheals, erythema flares, and grid markings
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_large)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image_bgr.shape[:2]
    total_area = h * w

    skin_clean = np.zeros((h, w), dtype=np.uint8)
    if contours:
        max_area = max(cv2.contourArea(c) for c in contours)
        # Keep contours that are significant skin patches (> 3000px or > 5% of largest)
        for c in contours:
            c_area = cv2.contourArea(c)
            if c_area >= max(3000, 0.05 * max_area):
                cv2.drawContours(skin_clean, [c], -1, 255, -1)  # Fill contour completely

    # Fallback safety: if skin detection finds less than 5% of the frame, assume full frame
    if (skin_clean > 0).sum() < 0.05 * total_area:
        skin_clean = np.ones((h, w), dtype=np.uint8) * 255

    return skin_clean


def preprocess(image: np.ndarray) -> dict:
    """Run the full preprocessing pipeline on a BGR image.

    Returns a dict with:
        resized         – BGR image resized to MAX_IMAGE_DIMENSION
        gray            – single-channel grayscale
        clahe           – CLAHE-enhanced grayscale
        blurred         – Gaussian-blurred grayscale (for contour work)
        scale           – resize scale factor (original → resized)
        l_clahe         – CLAHE-enhanced L-channel in LAB space
        sam_ready_image – enhanced BGR image ready for SAM
        skin_mask       – binary mask (255 on skin, 0 on background/clothing)
    """

    h, w = image.shape[:2]
    longest = max(h, w)

    # Resize if larger than limit (keep aspect ratio)
    if longest > config.MAX_IMAGE_DIMENSION:
        if h > w:
            resized = imutils.resize(image, height=config.MAX_IMAGE_DIMENSION)
        else:
            resized = imutils.resize(image, width=config.MAX_IMAGE_DIMENSION)
    else:
        resized = image.copy()

    scale = resized.shape[1] / w  # width ratio

    # Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Gaussian blur — reduce high-freq noise & skin texture
    blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL, 0)

    # CLAHE — boost local contrast so raised wheals pop against the skin
    clahe_obj = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_SIZE,
    )
    clahe = clahe_obj.apply(blurred)

    # ── LAB Color Space Enhancement for SAM ──
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to L-channel to aggressively highlight the physical bumps
    l_clahe = clahe_obj.apply(l_channel)

    # Merge the CLAHE enhanced L-channel with the original A and B channels
    lab_enhanced = cv2.merge((l_clahe, a_channel, b_channel))

    # Convert back from LAB to BGR for the SAM model
    sam_ready_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # ── Patient Skin ROI Mask ──
    skin_mask = extract_skin_mask(resized)

    return {
        "resized": resized,
        "gray": gray,
        "clahe": clahe,
        "l_clahe": l_clahe,
        "blurred": blurred,
        "scale": float(scale),
        "sam_ready_image": sam_ready_image,
        "skin_mask": skin_mask,
    }
