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
        print(f"[SAM] Checkpoint not found at {checkpoint}. Attempting auto-download...")
        try:
            from scripts.download_sam import main as download_sam_main
            download_sam_main()
        except Exception as e:
            print(f"[SAM] Auto-download failed: {e}")

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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
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
    prep : dict
        Preprocessed image dictionary containing 'sam_ready_image' and 'l_clahe'.
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
    H, W = image_rgb.shape[:2]

    # Optionally compute a bounding rect around the ArUco marker to exclude it
    aruco_rect = None
    if marker_corners is not None:
        x, y, w, h = cv2.boundingRect(marker_corners.astype(np.int32))
        aruco_rect = (x, y, x + w, y + h)

    # ── 1. Find Prompt Points via Multi-scale LoG ──
    # blob_log returns array of [y, x, sigma]
    from skimage.feature import blob_log
    blobs = blob_log(prep["l_clahe"], min_sigma=2, max_sigma=25, num_sigma=10, threshold=0.045)

    pts = [(float(b[1]), float(b[0]), float(b[2])) for b in blobs]

    # Spatial clustering (suppress point duplicates within 11px while prioritizing strong response)
    # Border margin filter: exclude image boundary artifacts (e.g., photo edge cutoffs)
    border_margin = 12
    candidates = []
    for p in sorted(pts, key=lambda item: -item[2]):
        cx, cy = p[0], p[1]
        if cx < border_margin or cx > (W - border_margin) or cy < border_margin or cy > (H - border_margin):
            continue
        # Exclude ArUco marker region
        if aruco_rect is not None:
            x1, y1, x2, y2 = aruco_rect
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                continue
        if not any(np.hypot(cx - c[0], cy - c[1]) < 11 for c in candidates):
            candidates.append((cx, cy))

    # Cap maximum candidates to prevent excessive CPU runtime on very noisy images
    if len(candidates) > 200:
        candidates = candidates[:200]

    if not candidates:
        return []

    # ── 2. Batched Inference via predict_torch ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    coords_np = np.array(candidates)[:, np.newaxis, :]  # shape: (B, 1, 2)
    labels_np = np.ones((len(candidates), 1), dtype=np.int32)

    coords_tf = predictor.transform.apply_coords(coords_np, (H, W))
    coords_torch = torch.as_tensor(coords_tf, dtype=torch.float, device=device)
    labels_torch = torch.as_tensor(labels_np, dtype=torch.int, device=device)

    with torch.no_grad():
        masks_tensor, scores_tensor, _ = predictor.predict_torch(
            point_coords=coords_torch,
            point_labels=labels_torch,
            multimask_output=True,
        )

    # Area thresholds in pixels
    min_area_px = max(config.SAM_MIN_MASK_REGION_AREA, config.MIN_WHEAL_AREA_MM2 * (ppm ** 2))
    max_area_px = config.MAX_WHEAL_AREA_MM2 * (ppm ** 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw_results = []

    for i, center_pt in enumerate(candidates):
        best_candidate = None
        best_score = -1.0

        for j in range(3):
            score = float(scores_tensor[i, j])
            if score < config.SAM_PRED_IOU_THRESH:
                continue

            mask_binary = masks_tensor[i, j].cpu().numpy().astype(np.uint8)
            area_px = float(np.sum(mask_binary))

            # Discard whole-arm/background masks or tiny speckles
            if area_px < min_area_px or area_px > max_area_px or area_px > 3000.0:
                continue

            # Smooth mask using shape prior
            mask_smooth = cv2.morphologyEx(mask_binary * 255, cv2.MORPH_CLOSE, kernel)
            mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            c_area = float(cv2.contourArea(contour))
            if c_area < min_area_px:
                continue

            # Shape filter (circularity and aspect ratio)
            if not _is_wheal_shaped(contour, c_area):
                continue

            # Measure diameter
            (_, _), radius = cv2.minEnclosingCircle(contour)
            diameter_px = radius * 2
            diameter_mm = diameter_px / ppm

            if diameter_mm < 1.0 or diameter_mm > 40.0:
                continue

            if score > best_score:
                best_score = score
                area_mm2 = c_area / (ppm ** 2)
                severity = _classify_severity(diameter_mm)
                best_candidate = {
                    "mask": mask_binary,
                    "contour": contour,
                    "center": center_pt,
                    "radius": radius,
                    "diameter_px": diameter_px,
                    "diameter_mm": diameter_mm,
                    "area_px": c_area,
                    "area_mm2": area_mm2,
                    "confidence": score,
                    "severity": severity,
                }

        if best_candidate is not None:
            raw_results.append(best_candidate)

    # ── 3. Mask-IoU & Adaptive Centroid NMS ──
    # Sort by confidence descending
    raw_results.sort(key=lambda item: -item["confidence"])
    deduped_candidates = []

    for cand in raw_results:
        is_dup = False
        for kept in deduped_candidates:
            # Overlap IoU check
            intersection = np.logical_and(cand["mask"], kept["mask"]).sum()
            union = np.logical_or(cand["mask"], kept["mask"]).sum()
            if union > 0 and (intersection / union) > 0.30:
                is_dup = True
                break
            # Close centroid check (relative to wheal radius)
            dist = np.hypot(cand["center"][0] - kept["center"][0],
                            cand["center"][1] - kept["center"][1])
            if dist < min(cand["radius"], kept["radius"]) * 0.75 and dist < 18.0:
                is_dup = True
                break

        if not is_dup:
            deduped_candidates.append(cand)

    # Sort by position: top-to-bottom, then left-to-right (for grid mapping)
    deduped_candidates.sort(key=lambda w: (w["center"][1], w["center"][0]))

    results: List[WhealResult] = []
    for wid, w in enumerate(deduped_candidates, start=1):
        results.append(WhealResult(
            id=wid,
            contour=w["contour"],
            center=w["center"],
            diameter_px=w["diameter_px"],
            diameter_mm=w["diameter_mm"],
            area_px=w["area_px"],
            area_mm2=w["area_mm2"],
            confidence=w["confidence"],
            severity=w["severity"],
        ))

    return results

