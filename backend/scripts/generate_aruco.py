#!/usr/bin/env python3
"""Generate a printable ArUco marker for calibration.

Usage:
    python -m backend.scripts.generate_aruco --id 0 --size 200 --output aruco_marker.png

The output PNG should be printed at EXACT scale (20mm × 20mm physical).
The script embeds DPI metadata so that common printers produce the correct size.
"""

import argparse
import cv2
import numpy as np


def generate_marker(
    marker_id: int = 0,
    pixel_size: int = 200,
    border_bits: int = 1,
    dict_type: int = cv2.aruco.DICT_4X4_50,
    physical_mm: float = 20.0,
    output_path: str = "aruco_marker.png",
) -> None:
    """Generate and save an ArUco marker with a white border."""

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, pixel_size)

    # Add a white border (25% of marker size on each side) for reliable detection
    border_px = pixel_size // 4
    bordered = cv2.copyMakeBorder(
        marker_img,
        border_px, border_px, border_px, border_px,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    # Calculate DPI so the marker prints at the correct physical size
    total_px = pixel_size + 2 * border_px
    total_mm = physical_mm * (total_px / pixel_size)  # Scale border proportionally
    total_inches = total_mm / 25.4
    dpi = int(round(total_px / total_inches))

    # Save as PNG with DPI metadata
    # OpenCV doesn't embed DPI, so we use a manual approach with PIL if available
    try:
        from PIL import Image
        pil_img = Image.fromarray(bordered)
        pil_img.save(output_path, dpi=(dpi, dpi))
    except ImportError:
        cv2.imwrite(output_path, bordered)
        print(f"[NOTE] Pillow not found — saved without DPI metadata. Print at {dpi} DPI to get {physical_mm}mm marker.")

    print(f"✅  ArUco marker saved to: {output_path}")
    print(f"    Dictionary : DICT_4X4_50")
    print(f"    Marker ID  : {marker_id}")
    print(f"    Pixel size : {pixel_size}px (marker) / {total_px}px (with border)")
    print(f"    Physical   : {physical_mm}mm × {physical_mm}mm (marker only)")
    print(f"    Print DPI  : {dpi}")
    print(f"\n    ⚠️  Print this at EXACT scale. Do NOT 'fit to page'.")
    print(f"    ⚠️  Place on the patient's back near the test site before photographing.")


def main():
    parser = argparse.ArgumentParser(description="Generate a printable ArUco calibration marker")
    parser.add_argument("--id", type=int, default=0, help="Marker ID (default: 0)")
    parser.add_argument("--size", type=int, default=200, help="Marker size in pixels (default: 200)")
    parser.add_argument("--physical-mm", type=float, default=20.0, help="Physical marker size in mm (default: 20)")
    parser.add_argument("--output", type=str, default="aruco_marker.png", help="Output file path")
    args = parser.parse_args()

    generate_marker(
        marker_id=args.id,
        pixel_size=args.size,
        physical_mm=args.physical_mm,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
