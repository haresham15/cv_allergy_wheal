"""Benchmark and evaluation script for allergy wheal detection on test photos.

Compares detected wheals against ground truth annotations in Testphotos/.
Computes clinical-grade computer vision metrics:
  - Precision, Recall, and F1 Score (detection rate)
  - Mean Absolute Error (MAE) and Percent Error in wheal diameter (mm)
  - Mask-level Intersection over Union (IoU) and Dice coefficient
"""

import os
import sys
import time
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.services.preprocessing import preprocess
from backend.services.segmentation import find_wheals
from backend.services.calibration import get_calibration
from backend.models.unet_rgbd import LateFusionUNet


def compute_mask_metrics(pred_mask: np.ndarray, target_mask: np.ndarray):
    """Compute binary IoU and Dice coefficient between two masks."""
    p = (pred_mask > 0).astype(np.uint8)
    t = (target_mask > 0).astype(np.uint8)
    intersection = int((p * t).sum())
    union = int(p.sum() + t.sum() - intersection)
    iou = 1.0 if union == 0 else float(intersection / union)
    dice = 1.0 if (p.sum() + t.sum()) == 0 else float(2 * intersection / (p.sum() + t.sum()))
    return iou, dice


def match_detections(gt_wheals: list, pred_wheals: list, max_dist_px: float = 18.0):
    """Match predicted wheals to ground truth wheals greedily by centroid distance.

    Returns:
        matches: list of (gt_idx, pred_idx, dist, gt_diam, pred_diam)
        tp: true positives count
        fp: false positives count
        fn: false negatives count
    """
    matched_gt = set()
    matched_pred = set()
    matches = []

    # Sort all pairs by distance
    pairs = []
    for gi, g in enumerate(gt_wheals):
        for pi, p in enumerate(pred_wheals):
            dist = np.hypot(g["center"][0] - p["center"][0], g["center"][1] - p["center"][1])
            if dist <= max(max_dist_px, g["radius"]):
                pairs.append((dist, gi, pi))

    pairs.sort(key=lambda x: x[0])

    for dist, gi, pi in pairs:
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
            matches.append({
                "gt_idx": gi,
                "pred_idx": pi,
                "dist": dist,
                "gt_diam_mm": gt_wheals[gi]["diameter_mm"],
                "pred_diam_mm": pred_wheals[pi]["diameter_mm"],
                "diam_error_mm": abs(gt_wheals[gi]["diameter_mm"] - pred_wheals[pi]["diameter_mm"]),
                "diam_pct_error": abs(gt_wheals[gi]["diameter_mm"] - pred_wheals[pi]["diameter_mm"]) / gt_wheals[gi]["diameter_mm"] * 100.0,
            })

    tp = len(matches)
    fp = len(pred_wheals) - tp
    fn = len(gt_wheals) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "matches": matches,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate():
    print("=" * 65)
    print(" [CLINICAL ALLERGY WHEAL DETECTION BENCHMARK]")
    print("=" * 65)

    # 1. Load Test Photo
    img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Testphotos", "allergy-Testing.jpg"))
    if not os.path.exists(img_path):
        print(f"[!] Test image not found at {img_path}")
        return 100.0

    image = cv2.imread(img_path)
    if image is None:
        print("[!] Failed to read test image")
        return 100.0

    # 2. Calibration & Preprocessing
    prep = preprocess(image)
    cal = get_calibration(prep["resized"])
    ppm = cal.ppm
    H, W = prep["resized"].shape[:2]
    print(f"Image dimensions: {W}x{H} | Calibration: {ppm:.2f} px/mm ({cal.method})")

    # 3. Load Ground Truth Mask (All 49 wheals)
    manual_mask_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Testphotos", "allergy-Testing_mask.png"))
    if not os.path.exists(manual_mask_path):
        print(f"[!] Ground truth mask not found at {manual_mask_path}")
        return 100.0

    gt_mask_raw = cv2.imread(manual_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask_raw.shape[:2] != (H, W):
        gt_mask = cv2.resize(gt_mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        gt_mask = gt_mask_raw.copy()

    gt_mask_bin = (gt_mask > 127).astype(np.uint8)

    gt_contours, _ = cv2.findContours(gt_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_wheals = []
    for c in gt_contours:
        area = cv2.contourArea(c)
        if area >= 10:
            (cx, cy), r = cv2.minEnclosingCircle(c)
            diam_mm = (r * 2) / ppm
            gt_wheals.append({
                "center": (cx, cy),
                "radius": r,
                "diameter_mm": diam_mm,
                "area_px": area,
                "contour": c,
            })

    print(f"Ground Truth contains {len(gt_wheals)} allergy prick wheals.")

    # ═══════════════════════════════════════════════════════════════════
    # 4. Evaluate SAM Pipeline (Production Engine)
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- Evaluating Production SAM Segmentation Pipeline ---")
    t0 = time.time()
    sam_detections = find_wheals(prep, ppm=ppm)
    sam_time = time.time() - t0
    print(f"SAM execution time: {sam_time:.2f}s | Detections found: {len(sam_detections)}")

    pred_wheals_sam = []
    sam_composite_mask = np.zeros((H, W), dtype=np.uint8)
    for w in sam_detections:
        pred_wheals_sam.append({
            "id": w.id,
            "center": w.center,
            "radius": (w.diameter_px / 2.0),
            "diameter_mm": w.diameter_mm,
            "area_px": w.area_px,
            "severity": w.severity,
            "confidence": w.confidence,
            "contour": w.contour,
        })
        cv2.drawContours(sam_composite_mask, [w.contour], -1, 1, -1)

    sam_match_res = match_detections(gt_wheals, pred_wheals_sam)
    sam_iou, sam_dice = compute_mask_metrics(sam_composite_mask, gt_mask_bin)

    sam_mae = np.mean([m["diam_error_mm"] for m in sam_match_res["matches"]]) if sam_match_res["matches"] else 999.0
    sam_mape = np.mean([m["diam_pct_error"] for m in sam_match_res["matches"]]) if sam_match_res["matches"] else 100.0

    print(f"\nSAM Segmentation Performance:")
    print(f"  * Detected Wheals:     {len(sam_detections)} / {len(gt_wheals)} ground truth")
    print(f"  * True Positives (TP): {sam_match_res['tp']}")
    print(f"  * False Positives (FP):{sam_match_res['fp']}")
    print(f"  * False Negatives (FN):{sam_match_res['fn']}")
    print(f"  * Detection Recall:    {sam_match_res['recall']*100:.1f}%")
    print(f"  * Detection Precision: {sam_match_res['precision']*100:.1f}%")
    print(f"  * Detection F1-Score:  {sam_match_res['f1']*100:.1f}%")
    print(f"  * Mean Diameter MAE:   {sam_mae:.2f} mm")
    print(f"  * Mean Diameter Error: {sam_mape:.2f}%")
    print(f"  * Overall Mask IoU:    {sam_iou:.4f}")
    print(f"  * Mask Dice Coeff:     {sam_dice:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # 5. Evaluate U-Net Model (Edge / Fallback Model)
    # ═══════════════════════════════════════════════════════════════════
    unet_ckpt = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints", "final_rgbd_unet.pth"))
    if os.path.exists(unet_ckpt):
        print("\n--- Evaluating Edge Late-Fusion RGB-D U-Net Model ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LateFusionUNet(n_classes=1, bilinear=False).to(device)
        ckpt = torch.load(unet_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        rgb_256 = cv2.resize(cv2.cvtColor(prep["resized"], cv2.COLOR_BGR2RGB), (256, 256))
        rgb_tensor = torch.from_numpy((rgb_256.astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0).to(device)
        depth_mock = np.full((256, 256), 0.25, dtype=np.float32)
        depth_tensor = torch.from_numpy(np.clip((depth_mock - 0.1) / 0.4, 0, 1)[np.newaxis]).unsqueeze(0).to(device)

        with torch.no_grad():
            unet_preds = (torch.sigmoid(model(rgb_tensor, depth_tensor)) > 0.5).float().cpu().numpy()[0, 0]

        unet_mask_u8 = (unet_preds * 255).astype(np.uint8)
        unet_mask_orig = cv2.resize(unet_mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
        unet_cnts, _ = cv2.findContours(unet_mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pred_wheals_unet = []
        for c in unet_cnts:
            area = cv2.contourArea(c)
            if area >= 10:
                (cx, cy), r = cv2.minEnclosingCircle(c)
                pred_wheals_unet.append({
                    "center": (cx, cy),
                    "radius": r,
                    "diameter_mm": (r * 2) / ppm,
                    "area_px": area,
                    "contour": c,
                })

        unet_match_res = match_detections(gt_wheals, pred_wheals_unet)
        unet_iou, unet_dice = compute_mask_metrics(unet_mask_orig, gt_mask_bin)
        unet_mae = np.mean([m["diam_error_mm"] for m in unet_match_res["matches"]]) if unet_match_res["matches"] else 999.0

        print(f"U-Net Performance:")
        print(f"  * Detected Wheals:     {len(pred_wheals_unet)}")
        print(f"  * Detection Recall:    {unet_match_res['recall']*100:.1f}%")
        print(f"  * Detection Precision: {unet_match_res['precision']*100:.1f}%")
        print(f"  * Detection F1-Score:  {unet_match_res['f1']*100:.1f}%")
        print(f"  * Mean Diameter MAE:   {unet_mae:.2f} mm")
        print(f"  * Mask IoU:            {unet_iou:.4f}")

    print("\n" + "=" * 65)
    print(f"BENCHMARK SUMMARY:")
    print(f"  * Production Engine (SAM): {sam_match_res['recall']*100:.1f}% recall, {sam_match_res['precision']*100:.1f}% precision, {sam_mae:.2f}mm MAE")
    print(f"  * Mean Measurement Error:  {sam_mape:.2f}%")
    print("=" * 65)


    return sam_mape


if __name__ == "__main__":
    evaluate()
