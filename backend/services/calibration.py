"""ArUco marker–based calibration for pixel → millimetre conversion.

Detects a printed ArUco marker in the image, measures its side-length in
pixels, and uses the known physical size (config.MARKER_SIZE_MM) to derive
a reliable pixels-per-millimetre (PPM) ratio.

Falls back to a rough image-dimension estimate when no marker is detected.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from core import config


@dataclass
class CalibrationResult:
    """Holds everything the rest of the pipeline needs about scale."""
    detected: bool
    ppm: float                                     # pixels per millimetre
    method: str                                    # "aruco" | "estimated"
    marker_corners: Optional[np.ndarray] = None    # 4 corner points (if detected)
    marker_id: Optional[int] = None
    body_region: Optional[str] = None              # "forearm" | "back" | "torso"
    warning: Optional[str] = None
    needs_confirmation: bool = False


def detect_aruco_marker(image: np.ndarray) -> Optional[CalibrationResult]:
    """Try to find an ArUco marker and compute PPM from it.

    Uses the modern ArucoDetector API (OpenCV ≥ 4.7).
    Returns a CalibrationResult on success, or None.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT_TYPE)
    params = cv2.aruco.DetectorParameters()

    # Tune for small markers on skin (slightly relax defaults)
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    # Use the first detected marker
    marker_corners = corners[0][0]  # shape (4, 2) — four x,y corner points
    marker_id = int(ids[0][0])

    # Compute side length in pixels (average of the four sides for robustness)
    side_lengths = []
    for i in range(4):
        p1 = marker_corners[i]
        p2 = marker_corners[(i + 1) % 4]
        side_lengths.append(np.linalg.norm(p2 - p1))

    avg_side_px = float(np.mean(side_lengths))
    ppm = avg_side_px / config.MARKER_SIZE_MM

    return CalibrationResult(
        detected=True,
        ppm=ppm,
        method="aruco",
        marker_corners=marker_corners,
        marker_id=marker_id,
        body_region="calibrated",
        warning=None,
        needs_confirmation=False,
    )


def _estimate_ppm(
    image: np.ndarray,
    skin_mask: Optional[np.ndarray] = None,
    body_location: Optional[str] = None,
) -> CalibrationResult:
    """Anatomical fallback when no physical calibration marker is present.
    
    Adjusts assumed physical skin width based on whether the test is on a forearm
    (~75mm) or a broad back/torso (~320mm).
    """
    h, w = image.shape[:2]

    # Resolve body location
    if body_location in ("back", "torso"):
        assumed_skin_width_mm = 320.0
        region = "back"
    elif body_location in ("forearm", "arm"):
        assumed_skin_width_mm = 75.0
        region = "forearm"
    else:
        # Automatic anatomical heuristic:
        # Forearm close-ups are taken inches away; skin occupies the whole frame with virtually no background.
        # Back/torso shots are taken from a distance to fit the back/shoulders, leaving background margins (top/sides).
        region = "forearm"
        assumed_skin_width_mm = 75.0

        if skin_mask is not None and (skin_mask > 0).any():
            rows_with_skin = np.where(skin_mask.any(axis=1))[0]
            cols_with_skin = np.where(skin_mask.any(axis=0))[0]

            if len(rows_with_skin) > 0 and len(cols_with_skin) > 0:
                margin_top = rows_with_skin[0] / h
                margin_bottom = (h - 1 - rows_with_skin[-1]) / h
                margin_left = cols_with_skin[0] / w
                margin_right = (w - 1 - cols_with_skin[-1]) / w

                # Check corner occupancy (back shots typically leave top corners as wall/background)
                corner_size = max(8, int(min(h, w) * 0.08))
                tl_empty = (skin_mask[:corner_size, :corner_size] > 0).mean() < 0.15
                tr_empty = (skin_mask[:corner_size, -corner_size:] > 0).mean() < 0.15

                # Large vertical skin presence (patient torso)
                skin_span_h = (rows_with_skin[-1] - rows_with_skin[0]) / h
                skin_span_w = (cols_with_skin[-1] - cols_with_skin[0]) / w

                # If there is meaningful background margin above or beside the body, or top corners are open room/wall:
                is_wide_back_shot = (
                    (margin_top >= 0.035 or margin_left >= 0.04 or margin_right >= 0.04) and skin_span_h > 0.40
                ) or (
                    (tl_empty and tr_empty) and skin_span_h > 0.45 and skin_span_w > 0.45
                )

                if is_wide_back_shot:
                    assumed_skin_width_mm = 320.0
                    region = "back"

    # Assume the skin test field spans ~75% of the frame width
    fraction = 0.75
    ppm = (w * fraction) / assumed_skin_width_mm

    warning_msg = (
        f"No ArUco marker detected. Scale is estimated using {region} anatomical heuristic (~{assumed_skin_width_mm:.0f}mm). "
        "For clinical diagnostic accuracy, place a printed 20mm ArUco marker beside the test site."
    )

    return CalibrationResult(
        detected=False,
        ppm=float(ppm),
        method="estimated",
        body_region=region,
        warning=warning_msg,
        needs_confirmation=True,
    )


def get_calibration(
    image: np.ndarray,
    skin_mask: Optional[np.ndarray] = None,
    body_location: Optional[str] = None,
) -> CalibrationResult:
    """Main entry point — try ArUco detection; fall back to anatomical estimation."""
    result = detect_aruco_marker(image)
    if result is not None:
        return result
    return _estimate_ppm(image, skin_mask=skin_mask, body_location=body_location)
