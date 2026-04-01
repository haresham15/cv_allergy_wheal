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

from ..core import config


@dataclass
class CalibrationResult:
    """Holds everything the rest of the pipeline needs about scale."""
    detected: bool
    ppm: float                                     # pixels per millimetre
    method: str                                    # "aruco" | "estimated"
    marker_corners: Optional[np.ndarray] = None    # 4 corner points (if detected)
    marker_id: Optional[int] = None


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
    )


def _estimate_ppm(image: np.ndarray) -> CalibrationResult:
    """Rough fallback: assume the image captures ~70mm of skin width."""
    h, w = image.shape[:2]
    assumed_skin_width_mm = 70.0
    fraction = 0.7
    ppm = (w * fraction) / assumed_skin_width_mm
    return CalibrationResult(detected=False, ppm=float(ppm), method="estimated")


def get_calibration(image: np.ndarray) -> CalibrationResult:
    """Main entry point — try ArUco detection; fall back to estimation."""
    result = detect_aruco_marker(image)
    if result is not None:
        return result
    return _estimate_ppm(image)
