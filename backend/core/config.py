"""Configuration constants for the backend."""

import os, cv2

# ─── Upload Constraints ──────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png"]
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB (SAM needs decent resolution)

# ─── ArUco Calibration ───────────────────────────────────────────────
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
MARKER_SIZE_MM = 20.0  # Physical side length of the printed ArUco marker

# ─── Image Preprocessing ─────────────────────────────────────────────
MAX_IMAGE_DIMENSION = 1024       # Resize longest side to this before processing
GAUSSIAN_BLUR_KERNEL = (5, 5)
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = (8, 8)

# ─── Wheal Detection ─────────────────────────────────────────────────
MIN_WHEAL_AREA_MM2 = 2.0         # Ignore blobs smaller than 2 mm²
MAX_WHEAL_AREA_MM2 = 2000.0      # Ignore blobs larger than 2000 mm²
MIN_CIRCULARITY = 0.4            # Wheals should be roughly circular
MAX_ASPECT_RATIO = 2.5           # Reject very elongated shapes

# ─── Severity Thresholds (wheal diameter in mm) ──────────────────────
SEVERITY_NORMAL_MAX = 3.0        # < 3 mm  → negative / normal
SEVERITY_MILD_MAX = 8.0          # 3–8 mm  → mild
                                 # ≥ 8 mm  → severe

# ─── SAM (Segment Anything Model) ────────────────────────────────────
SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "sam_vit_b_01ec64.pth",
)

# SAM Automatic Mask Generator parameters
SAM_POINTS_PER_SIDE = 64
SAM_PRED_IOU_THRESH = 0.85
SAM_STABILITY_SCORE_THRESH = 0.98
SAM_BOX_NMS_THRESH = 0.4
SAM_MIN_MASK_REGION_AREA = 100   # In pixels — pre-filter before mm conversion

# ─── Allergen Grid Defaults ──────────────────────────────────────────
DEFAULT_GRID_ROWS = 4
DEFAULT_GRID_COLS = 2
