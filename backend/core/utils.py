import base64
from datetime import datetime
import numpy as np
import cv2


def bytes_to_cv2_image(file_bytes: bytes):
    """Decode raw bytes into an OpenCV BGR image (numpy array).

    Raises ValueError if decoding fails.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def image_to_base64(image: np.ndarray, ext: str = "jpg") -> str:
    """Encode an OpenCV image (BGR) to a data URL (base64).

    Returns a string like: data:image/jpg;base64,.....
    """
    success, buf = cv2.imencode(f".{ext}", image)
    if not success:
        raise ValueError("Could not encode image to bytes")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"
