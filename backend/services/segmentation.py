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

from ..core import config

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
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

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

    _mask_generator = SamAutomaticMaskGenerator(
        model=_sam_model,
        points_per_side=config.SAM_POINTS_PER_SIDE,
        pred_iou_thresh=config.SAM_PRED_IOU_THRESH,
        stability_score_thresh=config.SAM_STABILITY_SCORE_THRESH,
        min_mask_region_area=config.SAM_MIN_MASK_REGION_AREA,
    )

    print("[SAM] Model loaded ✓")
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
    image: np.ndarray,
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

    mask_gen = _load_sam()

    # SAM expects RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_masks = mask_gen.generate(image_rgb)

    # Area thresholds in pixels (derived from mm² limits)
    min_area_px = config.MIN_WHEAL_AREA_MM2 * (ppm ** 2)
    max_area_px = config.MAX_WHEAL_AREA_MM2 * (ppm ** 2)

    # Optionally compute a bounding rect around the ArUco marker to exclude it
    aruco_rect = None
    if marker_corners is not None:
        x, y, w, h = cv2.boundingRect(marker_corners.astype(np.int32))
        aruco_rect = (x, y, x + w, y + h)

    results: List[WhealResult] = []
    wid = 1

    for sam_mask in sam_masks:
        mask_binary = sam_mask["segmentation"]  # bool ndarray
        area_px = float(sam_mask["area"])
        predicted_iou = float(sam_mask["predicted_iou"])
        stability = float(sam_mask["stability_score"])

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

        # ── Compute centroid ──
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        # ── Exclude regions overlapping the ArUco marker ──
        if aruco_rect is not None:
            x1, y1, x2, y2 = aruco_rect
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                continue

        # ── Measure ──
        (_, _), radius = cv2.minEnclosingCircle(contour)
        diameter_px = radius * 2
        diameter_mm = diameter_px / ppm
        area_mm2 = area_px / (ppm ** 2)

        severity = _classify_severity(diameter_mm)
        confidence = min(1.0, float(predicted_iou * stability))

        results.append(WhealResult(
            id=wid,
            contour=contour,
            center=(cx, cy),
            diameter_px=diameter_px,
            diameter_mm=diameter_mm,
            area_px=area_px,
            area_mm2=area_mm2,
            confidence=confidence,
            severity=severity,
        ))
        wid += 1

    # ── Centroid-based NMS: deduplicate overlapping detections ──
    # SAM often detects both the inner wheal AND the surrounding
    # erythema halo as separate masks at the same location.
    # Keep only the highest-confidence detection per location.
    min_dist_mm = 3.0  # wheals closer than 3mm apart are considered duplicates
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
