import cv2
import numpy as np

from ..core import config


def get_ppm(image: np.ndarray) -> float:
    """Estimate pixels-per-millimeter (ppm) from image dimensions.
    
    Without a calibration marker, we assume a typical allergy test skin area
    occupies ~60-80% of the image width. Standard allergy test areas are
    typically 5-10 cm wide.
    
    This is a fallback estimation; for best results, use a calibration marker.
    """
    
    height, width = image.shape[:2]
    # Assume image captures ~7 cm (70 mm) width of skin test area at 70% of image width
    assumed_skin_width_mm = 70.0
    image_width_fraction = 0.7
    
    ppm = (width * image_width_fraction) / assumed_skin_width_mm
    return float(ppm)


