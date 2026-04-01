#!/usr/bin/env python3
"""Download the SAM ViT-B model checkpoint.

Usage:
    python -m backend.scripts.download_sam

Downloads to backend/models/sam_vit_b_01ec64.pth (~375 MB).
"""

import os
import sys
import urllib.request

MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sam_vit_b_01ec64.pth")


def download_with_progress(url: str, dest: str) -> None:
    """Download a file with a simple progress indicator."""

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Downloading: {mb_down:.1f} / {mb_total:.1f} MB ({pct:.1f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✅  SAM model already exists at: {MODEL_PATH} ({size_mb:.1f} MB)")
        return

    print(f"📥  Downloading SAM ViT-B model...")
    print(f"    URL:  {MODEL_URL}")
    print(f"    Dest: {MODEL_PATH}")
    print()

    try:
        download_with_progress(MODEL_URL, MODEL_PATH)
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✅  Download complete! ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌  Download failed: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        sys.exit(1)


if __name__ == "__main__":
    main()
