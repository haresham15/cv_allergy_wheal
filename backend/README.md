---
title: WhealVision API
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🔬 WhealVision API & Web Demo

Clinical-grade segmentation and measurement of skin-prick allergy wheals using **Meta's Segment Anything Model (SAM)**.

## 🚀 Features
- **Accurate Sub-Millimeter Measurement:** Multi-scale LoG blob detection combined with SAM prompt-guided mask segmentation achieves sub-millimeter diameter precision (MAE < 1mm).
- **ZeroGPU Acceleration:** Dynamically accelerates inference with free Hugging Face ZeroGPU (Nvidia A100).
- **Dual Interface:** Serves an interactive Gradio web UI and a full REST API simultaneously.

## 🌐 API Reference

### `POST /api/v1/analyze`
Analyze an allergy test photo.

**Form Data:**
- `file`: Image file (JPEG or PNG)

**Response:**
```json
{
  "results": [
    {
      "id": 1,
      "center": [245, 310],
      "diameter_mm": 6.82,
      "area_mm2": 36.53,
      "severity": "mild",
      "confidence": 0.94
    }
  ],
  "meta": {
    "total_wheals": 44,
    "avg_diameter_mm": 5.42,
    "max_diameter_mm": 11.20,
    "severity_breakdown": {
      "negative": 0,
      "mild": 40,
      "severe": 4
    }
  },
  "calibration": {
    "pixels_per_mm": 8.42,
    "method": "aruco",
    "marker_detected": true
  },
  "visualization": {
    "annotated": "data:image/jpeg;base64,...",
    "segmented": "data:image/jpeg;base64,..."
  }
}
```

### `GET /health`
Returns service status and loaded model availability.
