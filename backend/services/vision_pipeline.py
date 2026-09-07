"""Full vision pipeline orchestrator.

Decode → Preprocess → Calibrate → Segment (SAM) → Map Allergens → Annotate → Return
"""

import cv2
import numpy as np
from typing import Dict, List, Optional

from core import utils, config
from . import preprocessing, calibration, segmentation, allergen_mapping


# Severity → colour map (BGR)
_SEVERITY_COLOURS = {
    "normal":  (0, 200, 0),     # green
    "mild":    (0, 220, 255),   # yellow / amber
    "severe":  (0, 0, 255),     # red
}


def _draw_annotations(
    image: np.ndarray,
    wheals: list,
    cal: calibration.CalibrationResult,
    ppm: float,
) -> np.ndarray:
    """Draw measurement overlays on a copy of the image."""

    annotated = image.copy()

    # ── Draw ArUco marker outline (blue) ──
    if cal.detected and cal.marker_corners is not None:
        pts = cal.marker_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], True, (255, 180, 0), 2, cv2.LINE_AA)
        cx = int(np.mean(cal.marker_corners[:, 0]))
        cy = int(np.mean(cal.marker_corners[:, 1])) - 12
        cv2.putText(annotated, f"ArUco ({config.MARKER_SIZE_MM:.0f}mm)",
                    (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 180, 0), 1, cv2.LINE_AA)

    # ── Draw each wheal ──
    for w in wheals:
        colour = (0, 0, 255) # red outline
        cx, cy = int(w.center[0]), int(w.center[1])

        # Draw contour
        cv2.drawContours(annotated, [w.contour], -1, colour, 2, cv2.LINE_AA)

        # Draw minimum enclosing circle (dashed feel via thinner line)
        (_, _), radius = cv2.minEnclosingCircle(w.contour)
        cv2.circle(annotated, (cx, cy), int(radius), colour, 1, cv2.LINE_AA)

        # Draw centre dot
        cv2.circle(annotated, (cx, cy), 3, colour, -1, cv2.LINE_AA)

    # ── Scale bar (bottom-right) ──
    bar_mm = 10.0
    bar_px = int(bar_mm * ppm)
    h, w_img = annotated.shape[:2]
    x1 = w_img - bar_px - 20
    y1 = h - 30
    cv2.line(annotated, (x1, y1), (x1 + bar_px, y1), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(annotated, (x1, y1 - 5), (x1, y1 + 5), (255, 255, 255), 2)
    cv2.line(annotated, (x1 + bar_px, y1 - 5), (x1 + bar_px, y1 + 5), (255, 255, 255), 2)
    cv2.putText(annotated, f"{bar_mm:.0f} mm", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


def _build_composite_mask(image: np.ndarray, wheals: list) -> np.ndarray:
    """Build a colour mask showing all detected wheal regions."""
    mask = np.zeros_like(image)
    for w in wheals:
        colour = _SEVERITY_COLOURS.get(w.severity, (200, 200, 200))
        cv2.drawContours(mask, [w.contour], -1, colour, -1)  # filled
        cv2.drawContours(mask, [w.contour], -1, (255, 255, 255), 1)  # outline
    return mask


try:
    import spaces
except ImportError:
    class _SpacesMock:
        @staticmethod
        def GPU(func=None, duration=None):
            if func is not None:
                return func
            def decorator(f):
                return f
            return decorator
    spaces = _SpacesMock()


@spaces.GPU(duration=120)
def process_image(
    file_bytes: bytes,
    allergen_grid: Optional[Dict[str, str]] = None,
    body_location: Optional[str] = None,
) -> dict:
    """Orchestrate the full processing pipeline.

    Parameters
    ----------
    file_bytes : bytes
        Raw uploaded image bytes (JPEG/PNG).
    allergen_grid : dict, optional
        Allergen mapping, e.g. {"A1": "Peanut", "A2": "Dust Mite"}.
    body_location : str, optional
        Anatomical region ("forearm" or "back") to refine calibration when no marker is present.

    Returns
    -------
    dict ready for JSON serialisation.
    """

    # 1. Decode
    img = utils.bytes_to_cv2_image(file_bytes)

    # 2. Preprocess (generates resized, CLAHE, and patient skin ROI mask)
    prep = preprocessing.preprocess(img)
    resized = prep["resized"]
    skin_mask = prep.get("skin_mask")

    # 3. Calibrate (ArUco detection or anatomical body-region fallback)
    cal = calibration.get_calibration(resized, skin_mask=skin_mask, body_location=body_location)
    ppm = cal.ppm

    # 4. Segment with SAM (skin ROI filtered, contrast ranked, calibrated area bounds)
    wheals = segmentation.find_wheals(
        prep, ppm,
        marker_corners=cal.marker_corners,
        cal_detected=cal.detected,
    )

    # 5. Map allergens (if grid supplied)
    if allergen_grid:
        labels, n_rows, n_cols = allergen_mapping.parse_grid_input(allergen_grid)
        if labels:
            h, w = resized.shape[:2]
            allergen_mapping.assign_allergens(
                wheals, labels, n_rows, n_cols, w, h,
            )

    # 6. Annotate
    annotated = _draw_annotations(resized, wheals, cal, ppm)
    mask_img = _build_composite_mask(resized, wheals)

    # 7. Encode images
    annotated_b64 = utils.image_to_base64(annotated)
    segmented_b64 = utils.image_to_base64(mask_img)

    # 8. Build response
    results = []
    for w in wheals:
        results.append({
            "id": w.id,
            "allergen": getattr(w, "allergen", None),
            "grid_position": getattr(w, "grid_position", None),
            "diameter_px": round(w.diameter_px, 2),
            "diameter_mm": round(w.diameter_mm, 2),
            "area_px": round(w.area_px, 2),
            "area_mm2": round(w.area_mm2, 2),
            "severity": w.severity,
            "confidence": round(w.confidence, 3),
            "center": [int(w.center[0]), int(w.center[1])],
        })

    # Summary stats
    diameters = [r["diameter_mm"] for r in results]
    severity_counts = {"normal": 0, "mild": 0, "severe": 0}
    for r in results:
        severity_counts[r["severity"]] = severity_counts.get(r["severity"], 0) + 1

    return {
        "meta": {
            "processed_at": utils.now_iso(),
            "total_wheals": len(results),
            "avg_diameter_mm": round(float(np.mean(diameters)), 2) if diameters else 0.0,
            "max_diameter_mm": round(float(max(diameters)), 2) if diameters else 0.0,
            "severity_breakdown": severity_counts,
            "image_width": resized.shape[1],
            "image_height": resized.shape[0],
        },
        "calibration": {
            "detected": cal.detected,
            "method": cal.method,
            "scale_ppm": round(cal.ppm, 4),
            "marker_id": cal.marker_id,
            "body_region": cal.body_region,
            "warning": cal.warning,
            "needs_confirmation": getattr(cal, "needs_confirmation", not cal.detected),
        },
        "results": results,
        "visualization": {
            "annotated": annotated_b64,
            "segmented": segmented_b64,
        },
    }
