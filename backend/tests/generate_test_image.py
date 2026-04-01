"""Generate a synthetic test image with an ArUco marker and simulated wheals.

This creates a realistic-ish test image: skin-toned background with circular
darker spots (simulated wheals) and an ArUco marker for calibration.
"""

import cv2
import numpy as np
import os

def create_test_image(output_path: str = "test_image.jpg"):
    # Create a skin-toned background (800x600)
    h, w = 600, 800
    image = np.full((h, w, 3), (160, 180, 210), dtype=np.uint8)  # BGR skin-ish tone
    
    # Add subtle noise for realism
    noise = np.random.normal(0, 8, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Slight gradient to simulate uneven lighting
    for y in range(h):
        factor = 0.85 + 0.3 * (y / h)
        image[y] = np.clip(image[y] * factor, 0, 255).astype(np.uint8)

    # --- Place ArUco marker (top-right corner) ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_px = 80  # pixels (represents 20mm physical)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, marker_px)
    # Add white border
    border = 10
    marker_bordered = cv2.copyMakeBorder(marker_img, border, border, border, border,
                                          cv2.BORDER_CONSTANT, value=255)
    # Convert to BGR
    marker_bgr = cv2.cvtColor(marker_bordered, cv2.COLOR_GRAY2BGR)
    
    mx, my = w - marker_bordered.shape[1] - 30, 30
    mh, mw = marker_bgr.shape[:2]
    image[my:my+mh, mx:mx+mw] = marker_bgr

    # --- Place simulated wheals in a grid pattern ---
    # 4 rows x 2 cols grid of test spots
    wheals = [
        # (cx, cy, radius, intensity_drop) - bigger radius = more severe
        (200, 120, 22, 40),   # A1: large wheal (severe - histamine control)
        (450, 120, 5, 15),    # A2: tiny/no wheal (negative control)
        (200, 230, 15, 30),   # B1: medium wheal (mild - dust mite)
        (450, 230, 18, 35),   # B2: medium-large wheal (mild-severe - cat)
        (200, 340, 12, 25),   # C1: small wheal (mild - dog)
        (450, 340, 20, 38),   # C2: large wheal (severe - peanut)
        (200, 450, 8, 20),    # D1: small wheal (normal - tree pollen)
        (450, 450, 14, 28),   # D2: medium wheal (mild - grass pollen)
    ]
    
    for cx, cy, radius, intensity in wheals:
        # Draw erythema (redness) around wheal
        erythema_radius = int(radius * 1.8)
        overlay = image.copy()
        cv2.circle(overlay, (cx, cy), erythema_radius, (130, 140, 220), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Draw wheal (raised, lighter/darker bump)
        overlay2 = image.copy()
        cv2.circle(overlay2, (cx, cy), radius, 
                   (160 - intensity, 180 - intensity, 210 - intensity//2), -1)
        cv2.addWeighted(overlay2, 0.6, image, 0.4, 0, image)
        
        # Add slight edge highlight (simulates raised skin)
        cv2.circle(image, (cx, cy), radius, 
                   (140 - intensity, 160 - intensity, 190 - intensity//2), 1)

    # Apply slight overall blur for realism
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"✅ Test image saved to: {output_path}")
    print(f"   Size: {w}x{h}")
    print(f"   ArUco marker: ID 0, {marker_px}px (= 20mm physical)")
    print(f"   Wheals: {len(wheals)} simulated spots in 4x2 grid")
    return output_path


if __name__ == "__main__":
    create_test_image()
