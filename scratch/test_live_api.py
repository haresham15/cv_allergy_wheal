import urllib.request
import json
import uuid
import os

image_path = "Testphotos/test_upload.webp"
if not os.path.exists(image_path):
    print(f"Error: {image_path} not found")
    exit(1)

boundary = uuid.uuid4().hex
with open(image_path, "rb") as f:
    img_data = f.read()

header = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="test_upload.webp"\r\n'
    f"Content-Type: image/webp\r\n\r\n"
).encode("utf-8")
footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
body = header + img_data + footer

url = "https://hareshm15-allergy-wheal-api.hf.space/api/v1/analyze"
req = urllib.request.Request(
    url,
    data=body,
    headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Origin": "https://cv-allergy-wheal.vercel.app",
        "User-Agent": "Mozilla/5.0",
    },
    method="POST",
)

print(f"Uploading {len(img_data)} bytes ({len(img_data)/(1024*1024):.2f} MB) to {url}...")
try:
    res = urllib.request.urlopen(req, timeout=120)
    print("Response Status Code:", res.status)
    data = json.loads(res.read())
    print("\n--- RESULTS FROM LIVE ZERO-GPU BACKEND ---")
    print(f"Total Wheals Detected: {data['meta']['total_wheals']}")
    print(f"Average Diameter: {data['meta']['avg_diameter_mm']:.2f} mm")
    print(f"Maximum Diameter: {data['meta']['max_diameter_mm']:.2f} mm")
    print(f"Severity Breakdown: {data['meta']['severity_breakdown']}")
    print(f"Visualization base64 length: {len(data['visualization']['annotated'])} chars")
    print("\nSUCCESS: Vision pipeline executed and returned clinical measurements!")
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code)
    print("Response body:", e.read().decode("utf-8", errors="ignore"))
except Exception as e:
    print("Request failed:", e)
