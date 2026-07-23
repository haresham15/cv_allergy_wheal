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


def preprocess(image: np.ndarray) -> dict:
    """Run the full preprocessing pipeline on a BGR image.

    Returns a dict with:
        resized   – BGR image resized to MAX_IMAGE_DIMENSION
        gray      – single-channel grayscale
        clahe     – CLAHE-enhanced grayscale
        blurred   – Gaussian-blurred grayscale (for contour work)
        scale     – resize scale factor (original → resized)
    """

    h, w = image.shape[:2]
    longest = max(h, w)

    # Resize if larger than limit (keep aspect ratio)
    if longest > config.MAX_IMAGE_DIMENSION:
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

    return {
        "resized": resized,
        "gray": gray,
        "clahe": clahe,
        "l_clahe": l_clahe,
        "blurred": blurred,
        "scale": float(scale),
        "sam_ready_image": sam_ready_image,
    }
