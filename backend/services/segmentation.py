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
    cal_detected: bool = False,
) -> List[WhealResult]:
    """Run SAM on *image* (BGR) and return filtered wheal detections.

    Parameters
    ----------
    prep : dict
        Preprocessed image dictionary containing 'sam_ready_image', 'l_clahe', and 'skin_mask'.
    ppm : float
        Pixels-per-millimetre from calibration.
    marker_corners : optional array
        If the ArUco marker was detected, pass its corners so we can
        exclude it from results.
    cal_detected : bool
        Whether scale was calibrated via physical ArUco marker or estimated.
    """

    predictor = _load_sam()

    # SAM expects RGB
    image_rgb = cv2.cvtColor(prep["sam_ready_image"], cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    H, W = image_rgb.shape[:2]
    skin_mask = prep.get("skin_mask")

    # Optionally compute a bounding rect around the ArUco marker to exclude it
    aruco_rect = None
    if marker_corners is not None:
        x, y, w, h = cv2.boundingRect(marker_corners.astype(np.int32))
        aruco_rect = (x, y, x + w, y + h)

    # ── 1. Find Prompt Points via Multi-scale LoG & Local Contrast ──
    from skimage.feature import blob_log
    blobs = blob_log(prep["l_clahe"], min_sigma=2, max_sigma=25, num_sigma=10, threshold=0.050)

    border_margin = 12
    scored_blobs = []
    for b in blobs:
        cy, cx, sigma = float(b[0]), float(b[1]), float(b[2])
        r, c = int(round(cy)), int(round(cx))

        # Boundary margin exclusion
        if r < border_margin or r > H - border_margin or c < border_margin or c > W - border_margin:
            continue

        # Patient skin ROI filter: exclude background, clothing, posters, jewelry, hair
        if skin_mask is not None and skin_mask[r, c] == 0:
            continue

        # Exclude ArUco marker region
        if aruco_rect is not None:
            x1, y1, x2, y2 = aruco_rect
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                continue

        # True local response contrast (ranking by signal magnitude instead of scale sigma)
        r0, r1 = max(0, r - 2), min(H, r + 3)
        c0, c1 = max(0, c - 2), min(W, c + 3)
        center_val = float(prep["l_clahe"][r0:r1, c0:c1].mean())

        R_outer = int(round(sigma * 2))
        r_out0, r_out1 = max(0, r - R_outer), min(H, r + R_outer + 1)
        c_out0, c_out1 = max(0, c - R_outer), min(W, c + R_outer + 1)
        outer_val = float(prep["l_clahe"][r_out0:r_out1, c_out0:c_out1].mean())
        contrast = abs(center_val - outer_val)

        if contrast < 4.0:
            continue

        scored_blobs.append((contrast, (cx, cy)))

    # Sort candidates by true local contrast response (descending)
    scored_blobs.sort(key=lambda item: -item[0])

    # Spatial clustering (suppress point duplicates within 10px while prioritizing highest contrast)
    candidates = []
    for contrast, (cx, cy) in scored_blobs:
        if not any(np.hypot(cx - c[0], cy - c[1]) < 10 for c in candidates):
            candidates.append((cx, cy))

    # Generous capacity to ensure small punctate wheals are never truncated
    if len(candidates) > 260:
        candidates = candidates[:260]

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

    # Safe decoupled area limits (prevents uncalibrated PPM squared blowup)
    if cal_detected:
        min_area_px = max(15.0, config.MIN_WHEAL_AREA_MM2 * (ppm ** 2))
        max_area_px = min(4000.0, config.MAX_WHEAL_AREA_MM2 * (ppm ** 2))
    else:
        min_area_px = max(15.0, min(50.0, config.MIN_WHEAL_AREA_MM2 * (ppm ** 2)))
        max_area_px = min(3500.0, max(800.0, config.MAX_WHEAL_AREA_MM2 * (ppm ** 2)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw_results = []

    for i, center_pt in enumerate(candidates):
        best_candidate = None
        best_score = -1.0

        for j in range(3):
            score = float(scores_tensor[i, j])
            # Soft score floor (0.65) allowing subtle wheals through to shape analysis
            if score < 0.65:
                continue

            mask_binary = masks_tensor[i, j].cpu().numpy().astype(np.uint8)
            area_px = float(np.sum(mask_binary))

            # Discard whole-arm/background masks or tiny speckles
            if area_px < min_area_px or area_px > max_area_px:
                continue

            # Skin mask validation: mask must be primarily on patient skin
            if skin_mask is not None and area_px > 0:
                outside_skin = np.logical_and(mask_binary, skin_mask == 0).sum() / area_px
                if outside_skin > 0.25:
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

