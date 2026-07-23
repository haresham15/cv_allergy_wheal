import re

with open('backend/services/segmentation.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace imports
content = content.replace("from segment_anything import sam_model_registry, SamAutomaticMaskGenerator", 
                          "from segment_anything import sam_model_registry, SamPredictor\n    from skimage.feature import blob_log")

# Replace _load_sam
load_sam_old = """    _mask_generator = SamAutomaticMaskGenerator(
        model=_sam_model,
        points_per_side=config.SAM_POINTS_PER_SIDE,
        pred_iou_thresh=config.SAM_PRED_IOU_THRESH,
        stability_score_thresh=config.SAM_STABILITY_SCORE_THRESH,
        box_nms_thresh=config.SAM_BOX_NMS_THRESH,
        min_mask_region_area=config.SAM_MIN_MASK_REGION_AREA,
    )"""

load_sam_new = """    _mask_generator = SamPredictor(_sam_model)"""

content = content.replace(load_sam_old, load_sam_new)

# Replace find_wheals
find_wheals_old = """def find_wheals(
    image: np.ndarray,
    ppm: float,
    marker_corners: Optional[np.ndarray] = None,
) -> List[WhealResult]:"""

find_wheals_new = """def find_wheals(
    prep: dict,
    ppm: float,
    marker_corners: Optional[np.ndarray] = None,
) -> List[WhealResult]:"""

content = content.replace(find_wheals_old, find_wheals_new)

# Replace the inner body of find_wheals
# From docstring end to the deduplication step.
body_pattern = r'    mask_gen = _load_sam\(\).*?wid \+= 1'

body_new = """    predictor = _load_sam()

    # SAM expects RGB
    image_rgb = cv2.cvtColor(prep["sam_ready_image"], cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # ── 1. Find Prompt Points via LoG ──
    # blob_log returns array of [y, x, sigma]
    # We use the CLAHE enhanced L-channel where cysts are bright blobs.
    from skimage.feature import blob_log
    blobs = blob_log(prep["l_clahe"], min_sigma=3, max_sigma=30, num_sigma=10, threshold=0.1)

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
        wid += 1"""

content = re.sub(body_pattern, body_new, content, flags=re.DOTALL)

with open('backend/services/segmentation.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated segmentation.py")
