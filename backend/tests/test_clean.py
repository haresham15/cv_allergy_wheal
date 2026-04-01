"""Clean test runner that writes results to a readable file."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Force UTF-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
from backend.tests.generate_test_image import create_test_image

API_URL = "http://localhost:8000/api/v1/analyze"
ALLERGEN_GRID = {
    "A1": "Histamine (Control+)", "A2": "Saline (Control-)",
    "B1": "Dust Mite", "B2": "Cat Dander",
    "C1": "Dog Dander", "C2": "Peanut",
    "D1": "Tree Pollen", "D2": "Grass Pollen",
}

results_file = os.path.join(os.path.dirname(__file__), "test_results.json")
report = {"tests": [], "errors": [], "warnings": []}

def run_test(name, fn):
    print(f"Running: {name}...", flush=True)
    try:
        result = fn()
        report["tests"].append({"name": name, "status": "PASS", "details": result})
        print(f"  PASS: {name}")
    except AssertionError as e:
        report["tests"].append({"name": name, "status": "FAIL", "error": str(e)})
        report["errors"].append(f"{name}: {e}")
        print(f"  FAIL: {name} -> {e}")
    except Exception as e:
        report["tests"].append({"name": name, "status": "ERROR", "error": str(e)})
        report["errors"].append(f"{name}: {e}")
        print(f"  ERROR: {name} -> {e}")

def test_health():
    r = requests.get("http://localhost:8000/", timeout=5)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    return r.json()

def test_invalid_file():
    r = requests.post(API_URL, files={"file": ("test.txt", b"hello", "text/plain")}, timeout=10)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}: {r.text}"
    return {"status_code": r.status_code}

def test_malformed_grid():
    img_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
    with open(img_path, "rb") as f:
        r = requests.post(API_URL, files={"file": ("t.jpg", f, "image/jpeg")},
                          data={"allergen_grid": "not-json"}, timeout=30)
    assert r.status_code == 400, f"Expected 400 for bad JSON, got {r.status_code}: {r.text}"
    return {"status_code": r.status_code}

def test_analyze_no_grid():
    img_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
    with open(img_path, "rb") as f:
        start = time.time()
        r = requests.post(API_URL, files={"file": ("t.jpg", f, "image/jpeg")}, timeout=180)
        elapsed = time.time() - start

    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:300]}"
    data = r.json()

    # Validate structure
    for key in ["meta", "calibration", "results", "visualization"]:
        assert key in data, f"Missing '{key}' in response"

    # Validate calibration
    cal = data["calibration"]
    if not cal["detected"]:
        report["warnings"].append("ArUco marker NOT detected - using fallback estimation")

    # Validate viz
    viz = data["visualization"]
    assert viz.get("annotated", "").startswith("data:image"), "Missing annotated base64"
    assert viz.get("segmented", "").startswith("data:image"), "Missing segmented base64"

    # Validate results have no allergen (no grid was sent)
    for w in data["results"]:
        for field in ["id", "diameter_mm", "severity", "confidence", "center"]:
            assert field in w, f"Wheal missing '{field}'"

    return {
        "elapsed_s": round(elapsed, 1),
        "total_wheals": data["meta"]["total_wheals"],
        "avg_diameter_mm": data["meta"]["avg_diameter_mm"],
        "max_diameter_mm": data["meta"]["max_diameter_mm"],
        "severity_breakdown": data["meta"]["severity_breakdown"],
        "calibration_detected": cal["detected"],
        "calibration_method": cal["method"],
        "scale_ppm": cal["scale_ppm"],
        "results": data["results"],
    }

def test_analyze_with_grid():
    img_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
    with open(img_path, "rb") as f:
        start = time.time()
        r = requests.post(API_URL,
                          files={"file": ("t.jpg", f, "image/jpeg")},
                          data={"allergen_grid": json.dumps(ALLERGEN_GRID)},
                          timeout=180)
        elapsed = time.time() - start

    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:300]}"
    data = r.json()

    results = data["results"]
    allergens_assigned = [w for w in results if w.get("allergen")]

    return {
        "elapsed_s": round(elapsed, 1),
        "total_wheals": len(results),
        "allergens_assigned": len(allergens_assigned),
        "results": results,
    }

def test_oversized_file():
    big_data = b"\xff\xd8\xff\xe0" + b"\x00" * (11 * 1024 * 1024)
    r = requests.post(API_URL, files={"file": ("big.jpg", big_data, "image/jpeg")}, timeout=30)
    assert r.status_code == 413, f"Expected 413 for oversized, got {r.status_code}"
    return {"status_code": r.status_code}


def main():
    # Generate test image first
    img_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
    create_test_image(img_path)

    run_test("Health Check", test_health)
    run_test("Invalid File Type Rejection", test_invalid_file)
    run_test("Analyze Without Grid", test_analyze_no_grid)
    run_test("Analyze With Allergen Grid", test_analyze_with_grid)
    run_test("Malformed Grid Rejection", test_malformed_grid)
    run_test("Oversized File Rejection", test_oversized_file)

    # Write results to JSON
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nResults written to: {results_file}")
    print(f"Errors: {len(report['errors'])}")
    print(f"Warnings: {len(report['warnings'])}")

    return len(report["errors"])


if __name__ == "__main__":
    sys.exit(main())
