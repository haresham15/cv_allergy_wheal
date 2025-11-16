import cv2
import numpy as np

from ..core import utils
from . import calibration, segmentation


def process_image(file_bytes: bytes) -> dict:
    """Orchestrate the processing pipeline for an uploaded image.

    Steps: decode -> calibration (ppm estimation) -> segmentation (wheals) -> draw -> return JSON-friendly dict
    """
    # decode
    img = utils.bytes_to_cv2_image(file_bytes)

    # calibration (estimate ppm from image dimensions)
    ppm = calibration.get_ppm(img)

    # segmentation (returns wheals and binary mask)
    wheals, binary_mask = segmentation.find_wheals(img, ppm)

    # draw annotated image
    annotated = img.copy()
    results = []
    for w in wheals:
        ellipse = w["ellipse"]
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
        angle = int(ellipse[2])
        cv2.ellipse(annotated, center, axes, angle, 0, 360, (0, 255, 0), 2)
        cv2.circle(annotated, center, 2, (0, 255, 0), -1)

        results.append(
            {
                "id": int(w.get("id", 0)),
                "diameter_mm": float(round(w.get("diameter_mm", 0.0), 3)),
                "severity": (
                    "mild"
                    if 3.0 < w.get("diameter_mm", 0.0) < 8.0
                    else ("severe" if w.get("diameter_mm", 0.0) >= 8.0 else "normal")
                ),
                "center_point": [int(w["center_point"][0]), int(w["center_point"][1])],
                "confidence": float(w.get("confidence", 0.0)),
            }
        )

    # Convert binary mask to 3-channel for better visualization
    mask_colored = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    # Highlight detected regions in red
    mask_colored[binary_mask > 0] = [0, 0, 255]

    annotated_b64 = utils.image_to_base64(annotated)
    segmented_b64 = utils.image_to_base64(mask_colored)

    return {
        "meta": {"processed_at": utils.now_iso(), "image_quality_score": "unknown"},
        "calibration": {"detected": False, "scale_ppm": float(ppm), "method": "estimated_from_image_dimensions"},
        "results": results,
        "visualization": {
            "annotated": annotated_b64,
            "segmented": segmented_b64,
        },
    }


