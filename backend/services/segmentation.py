"""SAM-based wheal segmentation.

Uses Meta's Segment Anything Model (SAM) to produce masks of every
distinct region in the image, then filters them by size, shape, and
circularity to identify allergy wheals.

This approach is vastly more robust across different skin tones and
lighting conditions than traditional colour-threshold methods.
"""

import cv2
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional

from core import config

# Lazy-loaded globals — the model is heavy; load once and reuse.
_sam_model = None
_mask_generator = None


@dataclass
class WhealResult:
    """One detected wheal."""
    id: int
    contour: np.ndarray         # OpenCV contour
    center: tuple               # (cx, cy) in pixels
    diameter_px: float
    diameter_mm: float
    area_px: float
    area_mm2: float
    confidence: float           # 0-1, from SAM's predicted IoU
    severity: str               # "normal" | "mild" | "severe"
    allergen: Optional[str] = None


def _load_sam():
    """Lazy-load the SAM model and mask generator."""
    global _sam_model, _mask_generator

    if _mask_generator is not None:
        return _mask_generator

    import os
    from segment_anything import sam_model_registry, SamPredictor
    from skimage.feature import blob_log

    checkpoint = config.SAM_CHECKPOINT_PATH
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"SAM checkpoint not found at {checkpoint}. "
            f"Run: python -m backend.scripts.download_sam"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM] Loading {config.SAM_MODEL_TYPE} on {device}…")

    _sam_model = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=checkpoint)
    _sam_model.to(device=device)

    _mask_generator = SamPredictor(_sam_model)

    print("[SAM] Model loaded OK")
    return _mask_generator


def _classify_severity(diameter_mm: float) -> str:
    if diameter_mm < config.SEVERITY_NORMAL_MAX:
        return "normal"
    elif diameter_mm < config.SEVERITY_MILD_MAX:
        return "mild"
    else:
        return "severe"


def _mask_to_contour(mask: np.ndarray):
    """Extract the largest external contour from a binary mask."""
    mask_u8 = (mask * 255).astype(np.uint8)
    
    # ── Morphological Smoothing (Shape Priors) ──
    # Create an elliptical kernel since wheals are round/elliptical
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Closing: fills small holes inside the mask
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    # Opening: removes small noise attached to the edges, smoothing the boundary
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _is_wheal_shaped(contour: np.ndarray, area_px: float) -> bool:
    """Check circularity and aspect ratio to reject non-wheal shapes."""
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False

    circularity = 4 * np.pi * area_px / (perimeter ** 2)
    if circularity < config.MIN_CIRCULARITY:
        return False

    if len(contour) >= 5:
        _, (major, minor), _ = cv2.fitEllipse(contour)
        aspect = max(major, minor) / (min(major, minor) + 1e-6)
        if aspect > config.MAX_ASPECT_RATIO:
            return False

    return True


def find_wheals(
    prep: dict,
    ppm: float,
    marker_corners: Optional[np.ndarray] = None,
) -> List[WhealResult]:
    """Run SAM on *image* (BGR) and return filtered wheal detections.

    Parameters
    ----------
    image : np.ndarray
        BGR image (already resized by the preprocessing stage).
    ppm : float
        Pixels-per-millimetre from calibration.
    marker_corners : optional array
        If the ArUco marker was detected, pass its corners so we can
        exclude it from results.
    """

    predictor = _load_sam()

    # SAM expects RGB
    image_rgb = cv2.cvtColor(prep["sam_ready_image"], cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # ── 1. Find Prompt Points via LoG ──
    # blob_log returns array of [y, x, sigma]
    # We use the CLAHE enhanced L-channel where cysts are bright blobs.
    from skimage.feature import blob_log
    blobs = blob_log(prep["l_clahe"], min_sigma=3, max_sigma=30, num_sigma=10, threshold=0.05)

    # Area thresholds in pixels
    min_area_px = config.MIN_WHEAL_AREA_MM2 * (ppm ** 2)
    max_area_px = config.MAX_WHEAL_AREA_MM2 * (ppm ** 2)

    # Optionally compute a bounding rect around the ArUco marker to exclude it
    aruco_rect = None
    if marker_corners is not None:
        x, y, w, h = cv2.boundingRect(marker_corners.astype(np.int32))
        aruco_rect = (x, y, x + w, y + h)

    results: List[WhealResult] = []
    wid = 1

    for blob in blobs:
        y, x, r = blob
        cx, cy = float(x), float(y)

        # Exclude regions overlapping the ArUco marker before even prompting SAM
        if aruco_rect is not None:
            x1, y1, x2, y2 = aruco_rect
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                continue

        # ── 2. Prompt SAM ──
        input_point = np.array([[cx, cy]])
        input_label = np.array([1]) # foreground

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        mask_binary = masks[0]
        predicted_iou = float(scores[0])

        # Apply confidence threshold
        if predicted_iou < config.SAM_PRED_IOU_THRESH:
            continue

        area_px = float(np.sum(mask_binary))

        # ── Size filter ──
        if area_px < min_area_px or area_px > max_area_px:
            continue

        # ── Extract main contour ──
        contour = _mask_to_contour(mask_binary)
        if contour is None:
            continue

        # ── Shape filter ──
        if not _is_wheal_shaped(contour, area_px):
            continue

        # ── Measure ──
        (_, _), radius = cv2.minEnclosingCircle(contour)
        diameter_px = radius * 2
        diameter_mm = diameter_px / ppm
        
        # ── Strict Diameter Filtering ──
        if diameter_mm < 0.5 or diameter_mm > 40.0:
            continue

        area_mm2 = area_px / (ppm ** 2)
        severity = _classify_severity(diameter_mm)

        results.append(WhealResult(
            id=wid,
            contour=contour,
            center=(cx, cy),
            diameter_px=diameter_px,
            diameter_mm=diameter_mm,
            area_px=area_px,
            area_mm2=area_mm2,
            confidence=predicted_iou,
            severity=severity,
        ))
        wid += 1

    # Centroid-based NMS: deduplicate overlapping detections ──
    # SAM often detects both the inner wheal AND the surrounding
    # erythema halo as separate masks at the same location.
    # Keep only the highest-confidence detection per location.
    min_dist_mm = 8.0  # wheals closer than 8mm apart are considered duplicates
    min_dist_px = min_dist_mm * ppm
    deduped: List[WhealResult] = []
    for candidate in sorted(results, key=lambda w: -w.confidence):
        is_dup = False
        for kept in deduped:
            dist = np.hypot(candidate.center[0] - kept.center[0],
                            candidate.center[1] - kept.center[1])
            if dist < min_dist_px:
                is_dup = True
                break
        if not is_dup:
            deduped.append(candidate)

    # Sort by position: top-to-bottom, then left-to-right (for grid mapping)
    deduped.sort(key=lambda w: (w.center[1], w.center[0]))

    # Re-assign IDs after sorting
    for i, w in enumerate(deduped, start=1):
        w.id = i

    return deduped
