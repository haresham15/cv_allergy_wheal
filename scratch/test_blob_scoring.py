import os, sys, cv2, numpy as np
from skimage.feature import blob_log

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

try:
    from backend.services.preprocessing import preprocess
except ImportError:
    from services.preprocessing import preprocess

img = cv2.imread("Testphotos/allergy-Testing.jpg")
prep = preprocess(img)
l_clahe = prep["l_clahe"]
H, W = l_clahe.shape[:2]

blobs = blob_log(l_clahe, min_sigma=2, max_sigma=25, num_sigma=10, threshold=0.045)
print(f"Total blobs detected: {len(blobs)}")
print("Sample blobs (r, c, sigma):")
for b in blobs[:10]:
    print(f"  r={b[0]:.1f}, c={b[1]:.1f}, sigma={b[2]:.2f}")

# Check local contrast for each blob
scores = []
for b in blobs:
    r, c, s = int(round(b[0])), int(round(b[1])), int(round(b[2]))
    # Local patch
    r0, r1 = max(0, r - 2), min(H, r + 3)
    c0, c1 = max(0, c - 2), min(W, c + 3)
    center_val = float(l_clahe[r0:r1, c0:c1].mean())
    
    # Outer ring
    R_outer = int(round(s * 2))
    r_out0, r_out1 = max(0, r - R_outer), min(H, r + R_outer + 1)
    c_out0, c_out1 = max(0, c - R_outer), min(W, c + R_outer + 1)
    outer_val = float(l_clahe[r_out0:r_out1, c_out0:c_out1].mean())
    
    contrast = abs(center_val - outer_val)
    scores.append((contrast, b))

scores.sort(key=lambda x: -x[0])
print("\nTop 5 blobs by local contrast:")
for sc, b in scores[:5]:
    print(f"  contrast={sc:.2f}, r={b[0]:.1f}, c={b[1]:.1f}, sigma={b[2]:.2f}")
