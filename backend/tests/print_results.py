import json

with open(r'backend\tests\test_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 60)
print("  TEST RESULTS SUMMARY")
print("=" * 60)

for t in data['tests']:
    status_icon = "PASS" if t['status'] == 'PASS' else "FAIL"
    print(f"  [{status_icon}] {t['name']}")
    if 'details' in t and t['details']:
        d = t['details']
        # Print key details
        for key in ['elapsed_s', 'total_wheals', 'avg_diameter_mm', 'max_diameter_mm',
                     'severity_breakdown', 'calibration_detected', 'calibration_method',
                     'scale_ppm', 'allergens_assigned', 'status_code']:
            if key in d:
                print(f"         {key}: {d[key]}")
        # Print per-wheal results
        if 'results' in d and d['results']:
            print(f"         --- Wheal Details ---")
            for w in d['results']:
                allergen_str = w.get('allergen') or 'N/A'
                grid_str = w.get('grid_position') or 'N/A'
                print(f"         #{w['id']} | grid={grid_str} | allergen={allergen_str} | "
                      f"diameter={w['diameter_mm']}mm | severity={w['severity']} | "
                      f"confidence={w['confidence']}")
    if t['status'] != 'PASS':
        print(f"         ERROR: {t.get('error', 'unknown')}")

print(f"\nTotal Errors: {len(data['errors'])}")
print(f"Total Warnings: {len(data['warnings'])}")
if data['errors']:
    for e in data['errors']:
        print(f"  ERROR: {e}")
if data['warnings']:
    for w in data['warnings']:
        print(f"  WARNING: {w}")
