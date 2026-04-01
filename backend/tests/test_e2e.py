"""End-to-end test of the WhealVision backend API.

Generates a synthetic test image, sends it to the /api/v1/analyze endpoint,
and validates the response structure and content.
"""

import json
import sys
import os
import requests
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.tests.generate_test_image import create_test_image

API_URL = "http://localhost:8000/api/v1/analyze"

ALLERGEN_GRID = {
    "A1": "Histamine (Control+)",
    "A2": "Saline (Control-)",
    "B1": "Dust Mite",
    "B2": "Cat Dander",
    "C1": "Dog Dander",
    "C2": "Peanut",
    "D1": "Tree Pollen",
    "D2": "Grass Pollen",
}

ERRORS = []
WARNINGS = []


def log_error(msg):
    ERRORS.append(msg)
    print(f"  ❌ ERROR: {msg}")


def log_warning(msg):
    WARNINGS.append(msg)
    print(f"  ⚠️  WARNING: {msg}")


def log_ok(msg):
    print(f"  ✅ {msg}")


def main():
    print("=" * 60)
    print("  WhealVision — End-to-End API Test")
    print("=" * 60)

    # ─── Step 1: Generate test image ──────────────────────────
    print("\n📸 Step 1: Generate synthetic test image")
    test_img_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
    try:
        create_test_image(test_img_path)
        if os.path.exists(test_img_path):
            size_kb = os.path.getsize(test_img_path) / 1024
            log_ok(f"Test image created ({size_kb:.1f} KB)")
        else:
            log_error("Test image file not found after creation")
            return
    except Exception as e:
        log_error(f"Failed to create test image: {e}")
        return

    # ─── Step 2: Check backend health ────────────────────────
    print("\n🏥 Step 2: Health check")
    try:
        r = requests.get("http://localhost:8000/", timeout=5)
        if r.status_code == 200:
            log_ok(f"Backend healthy: {r.json()}")
        else:
            log_error(f"Health check failed: HTTP {r.status_code}")
            return
    except requests.ConnectionError:
        log_error("Cannot connect to backend on port 8000. Is it running?")
        return

    # ─── Step 3: Test with invalid file type ──────────────────
    print("\n🚫 Step 3: Test invalid file type rejection")
    try:
        r = requests.post(API_URL, files={"file": ("test.txt", b"hello", "text/plain")})
        if r.status_code == 400:
            log_ok(f"Correctly rejected invalid file type (HTTP 400)")
        else:
            log_error(f"Expected HTTP 400 for invalid file type, got {r.status_code}: {r.text}")
    except Exception as e:
        log_error(f"Invalid file type test failed: {e}")

    # ─── Step 4: Test without allergen grid ───────────────────
    print("\n🔬 Step 4: Analyze image (WITHOUT allergen grid)")
    try:
        with open(test_img_path, "rb") as f:
            start = time.time()
            r = requests.post(API_URL, files={"file": ("test.jpg", f, "image/jpeg")}, timeout=120)
            elapsed = time.time() - start

        print(f"     Response time: {elapsed:.1f}s")

        if r.status_code != 200:
            log_error(f"Analyze (no grid) failed: HTTP {r.status_code}")
            print(f"     Response body: {r.text[:500]}")
            return

        data = r.json()
        log_ok(f"Got response (HTTP 200)")

        # Validate response structure
        for key in ["meta", "calibration", "results", "visualization"]:
            if key in data:
                log_ok(f"Response has '{key}' field")
            else:
                log_error(f"Missing '{key}' field in response")

        # Validate meta
        meta = data.get("meta", {})
        print(f"     Total wheals: {meta.get('total_wheals', '?')}")
        print(f"     Avg diameter: {meta.get('avg_diameter_mm', '?')} mm")
        print(f"     Max diameter: {meta.get('max_diameter_mm', '?')} mm")
        print(f"     Severity breakdown: {meta.get('severity_breakdown', '?')}")

        # Validate calibration
        cal = data.get("calibration", {})
        print(f"     Calibration detected: {cal.get('detected', '?')}")
        print(f"     Calibration method: {cal.get('method', '?')}")
        print(f"     Scale PPM: {cal.get('scale_ppm', '?')}")

        if cal.get("detected"):
            log_ok("ArUco marker was detected!")
        else:
            log_warning("ArUco marker was NOT detected — using fallback estimation")

        # Validate results
        results = data.get("results", [])
        if len(results) > 0:
            log_ok(f"Detected {len(results)} wheals")
        else:
            log_warning("No wheals detected (may be expected depending on test image)")

        for w in results:
            for field in ["id", "diameter_mm", "severity", "confidence", "center"]:
                if field not in w:
                    log_error(f"Wheal result missing field: {field}")

            # Check allergen should be None when no grid supplied
            if w.get("allergen") is not None:
                log_warning(f"Wheal #{w.get('id')} has allergen set without grid input")

        # Validate visualization
        viz = data.get("visualization", {})
        if viz.get("annotated", "").startswith("data:image"):
            log_ok("Annotated image base64 present")
        else:
            log_error("Missing or invalid annotated image")

        if viz.get("segmented", "").startswith("data:image"):
            log_ok("Segmented mask base64 present")
        else:
            log_error("Missing or invalid segmented mask")

    except requests.Timeout:
        log_error("Request timed out (>120s) — SAM may be too slow")
    except Exception as e:
        log_error(f"Analyze (no grid) failed with exception: {e}")
        import traceback
        traceback.print_exc()

    # ─── Step 5: Test WITH allergen grid ──────────────────────
    print("\n🧪 Step 5: Analyze image (WITH allergen grid)")
    try:
        with open(test_img_path, "rb") as f:
            start = time.time()
            r = requests.post(
                API_URL,
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"allergen_grid": json.dumps(ALLERGEN_GRID)},
                timeout=120,
            )
            elapsed = time.time() - start

        print(f"     Response time: {elapsed:.1f}s")

        if r.status_code != 200:
            log_error(f"Analyze (with grid) failed: HTTP {r.status_code}")
            print(f"     Response body: {r.text[:500]}")
        else:
            data = r.json()
            log_ok(f"Got response with grid (HTTP 200)")

            results = data.get("results", [])
            allergens_assigned = [w for w in results if w.get("allergen")]
            if allergens_assigned:
                log_ok(f"{len(allergens_assigned)}/{len(results)} wheals have allergen labels")
                for w in allergens_assigned:
                    print(f"       #{w['id']} {w.get('grid_position', '?')} → "
                          f"{w['allergen']}: {w['diameter_mm']} mm ({w['severity']})")
            else:
                log_warning("No allergens were assigned to any wheals")

    except requests.Timeout:
        log_error("Request with grid timed out (>120s)")
    except Exception as e:
        log_error(f"Analyze (with grid) failed with exception: {e}")
        import traceback
        traceback.print_exc()

    # ─── Step 6: Test malformed allergen grid ─────────────────
    print("\n💥 Step 6: Test malformed allergen grid")
    try:
        with open(test_img_path, "rb") as f:
            r = requests.post(
                API_URL,
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"allergen_grid": "not-valid-json"},
                timeout=30,
            )
        if r.status_code == 400:
            log_ok("Correctly rejected malformed JSON grid (HTTP 400)")
        else:
            log_error(f"Expected HTTP 400 for malformed grid, got {r.status_code}")
    except Exception as e:
        log_error(f"Malformed grid test failed: {e}")

    # ─── Step 7: Test oversized file ──────────────────────────
    print("\n📦 Step 7: Test oversized file rejection")
    try:
        big_data = b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024)  # >10 MB fake JPEG header
        r = requests.post(API_URL, files={"file": ("big.jpg", big_data, "image/jpeg")}, timeout=30)
        if r.status_code == 413:
            log_ok("Correctly rejected oversized file (HTTP 413)")
        else:
            log_warning(f"Expected HTTP 413 for oversized file, got {r.status_code}")
    except Exception as e:
        log_error(f"Oversized file test failed: {e}")

    # ─── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    print(f"  Errors:   {len(ERRORS)}")
    print(f"  Warnings: {len(WARNINGS)}")

    if ERRORS:
        print("\n  ❌ ERRORS:")
        for e in ERRORS:
            print(f"     - {e}")
    if WARNINGS:
        print("\n  ⚠️  WARNINGS:")
        for w in WARNINGS:
            print(f"     - {w}")

    if not ERRORS:
        print("\n  🎉 All critical tests passed!")
    else:
        print(f"\n  💔 {len(ERRORS)} error(s) need fixing.")

    return len(ERRORS)


if __name__ == "__main__":
    sys.exit(main())
