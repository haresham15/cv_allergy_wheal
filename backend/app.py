"""Hugging Face Spaces Entry Point for WhealVision Backend.

Runs the FastAPI backend and provides an interactive Gradio interface
mounted on the same web server.
Supports both CPU Basic (16GB Free Tier) and ZeroGPU (Free Nvidia A100 Tier).
"""

import os
import sys
import io
import base64
import numpy as np
from PIL import Image

# Ensure backend root is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Auto-download SAM ViT-B model if not present on disk
try:
    from scripts.download_sam import main as download_sam_main
    download_sam_main()
except Exception as e:
    print(f"[Warning] Model auto-download check: {e}")

import gradio as gr
import uvicorn
from main import app as fastapi_app
from services.vision_pipeline import process_image

# Detect Hugging Face ZeroGPU if available
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False


def run_analysis(input_image):
    """Gradio handler: accepts numpy image, processes via vision_pipeline, returns visuals and JSON."""
    if input_image is None:
        return None, None, {"error": "Please upload a photo of the allergy test site."}

    # Convert image to JPEG bytes
    pil_img = Image.fromarray(input_image) if isinstance(input_image, np.ndarray) else input_image
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # Process using core vision pipeline
    result = process_image(img_bytes)

    # Decode base64 visualizations to PIL Images for Gradio display
    annotated_b64 = result["visualization"]["annotated"].split(",")[-1]
    segmented_b64 = result["visualization"]["segmented"].split(",")[-1]

    annotated_img = Image.open(io.BytesIO(base64.b64decode(annotated_b64)))
    segmented_img = Image.open(io.BytesIO(base64.b64decode(segmented_b64)))

    metrics_summary = {
        "processed_at": result["meta"]["processed_at"],
        "total_wheals_detected": result["meta"]["total_wheals"],
        "average_diameter_mm": result["meta"]["avg_diameter_mm"],
        "maximum_diameter_mm": result["meta"]["max_diameter_mm"],
        "severity_breakdown": result["meta"]["severity_breakdown"],
        "calibration": result["calibration"],
        "wheals": [
            {
                "id": w["id"],
                "diameter_mm": w["diameter_mm"],
                "area_mm2": w["area_mm2"],
                "severity": w["severity"],
                "confidence": w["confidence"],
                "center": w["center"],
            }
            for w in result["results"]
        ],
    }

    return annotated_img, segmented_img, metrics_summary


# Wrap with ZeroGPU decorator if running on Hugging Face Spaces with ZeroGPU
if HAS_SPACES:
    run_analysis = spaces.GPU(run_analysis)

# ── Build Gradio Interactive Web Interface ──
with gr.Blocks(title="WhealVision API & Web Demo") as demo:
    gr.Markdown(
        """
        # 🔬 WhealVision - Allergy Wheal Detection Backend
        AI-powered segmentation and measurement of skin-prick allergy wheals using **Meta's Segment Anything Model (SAM)**.
        Supports high-resolution images up to **50 MB** (JPEG, PNG, WebP, BMP, TIFF, HEIC).
        
        ### 🌐 API Access for Next.js & Clients:
        - **REST API Endpoint:** `POST /api/v1/analyze` (Multipart form with `file`)
        - **Health Check:** `GET /health`
        - **Interactive OpenAPI Documentation:** `GET /docs`
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Skin Prick Test Photo", type="numpy")
            btn = gr.Button("Analyze Wheals", variant="primary", size="lg")

        with gr.Column(scale=1):
            out_annotated = gr.Image(label="Annotated Measurements (Red Outlines)")
            out_segmented = gr.Image(label="Composite Wheal Mask")
            out_json = gr.JSON(label="Quantitative Analysis Results")

    btn.click(
        fn=run_analysis,
        inputs=[input_img],
        outputs=[out_annotated, out_segmented, out_json],
    )

# Mount Gradio onto the existing FastAPI application at root ("/")
# This allows BOTH the Gradio UI and all FastAPI REST endpoints (/api/v1/analyze, /health, /docs)
# to run concurrently on the same port!
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
