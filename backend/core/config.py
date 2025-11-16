"""Configuration constants for the backend."""

# Allowed upload content types
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png"]

# Max upload size in bytes (5 MB)
MAX_UPLOAD_SIZE = 5 * 1024 * 1024

# Marker size in millimeters (side length of ArUco marker used for calibration)
MARKER_SIZE_MM = 20.0

# Default minimum wheal area (mm^2) to ignore noise
MIN_WHEAL_AREA_MM2 = 2.0
