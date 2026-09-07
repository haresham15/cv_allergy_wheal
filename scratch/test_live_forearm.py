import urllib.request, json, uuid

boundary = uuid.uuid4().hex
with open('Testphotos/test_upload.webp', 'rb') as f:
    file_bytes = f.read()

parts = []
parts.append(b'--' + boundary.encode() + b'\r\n')
parts.append(b'Content-Disposition: form-data; name="file"; filename="test_upload.webp"\r\n')
parts.append(b'Content-Type: image/webp\r\n\r\n')
parts.append(file_bytes)
parts.append(b'\r\n--' + boundary.encode() + b'\r\n')
parts.append(b'Content-Disposition: form-data; name="body_location"\r\n\r\n')
parts.append(b'forearm\r\n')
parts.append(b'--' + boundary.encode() + b'--\r\n')
body = b''.join(parts)

req = urllib.request.Request('https://hareshm15-allergy-wheal-api.hf.space/api/v1/analyze', data=body)
req.add_header('Content-Type', 'multipart/form-data; boundary=' + boundary)

with urllib.request.urlopen(req, timeout=30) as resp:
    res = json.loads(resp.read().decode('utf-8'))
    print('Calibration (with forearm):', json.dumps(res.get('calibration'), indent=2))
    print('Meta:', json.dumps(res.get('meta'), indent=2))
