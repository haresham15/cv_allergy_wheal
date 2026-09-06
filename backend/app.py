"""Hugging Face Spaces Entry Point for WhealVision Backend.

Runs the FastAPI backend and provides an interactive Gradio interface
mounted on the same web server.
Supports both CPU Basic (16GB Free Tier) and ZeroGPU (Free Nvidia A100 Tier).
"""

# Hugging Face ZeroGPU support - MUST be imported before torch, gradio, or any other library!
try:
    import spaces
except ImportError:
    class _SpacesMock:
        @staticmethod
        def GPU(func=None, duration=None):
            if func is not None:
                return func
            def decorator(f):
                return f
            return decorator
    spaces = _SpacesMock()

@spaces.GPU()
def dummy_gpu_func():
    """Satisfies Hugging Face ZeroGPU static and startup analysis."""
    pass

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

# Compatibility shim for huggingface_hub v1.0+ removing HfFolder
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class _HfFolderShim:
            @staticmethod
            def get_token():
                try:
                    return huggingface_hub.get_token()
                except Exception:
                    return None
            @staticmethod
            def save_token(token):
                try:
                    huggingface_hub.login(token=token)
                except Exception:
                    pass
            @staticmethod
            def delete_token():
                try:
                    huggingface_hub.logout()
                except Exception:
                    pass
        huggingface_hub.HfFolder = _HfFolderShim
except Exception:
    pass

import gradio as gr
from gradio.routes import App
from fastapi.middleware.cors import CORSMiddleware
from routers.v1 import measurements
from main import health_check
from services.vision_pipeline import process_image

# ── 1. Create FastAPI Application Instance ──
custom_app = App(
    title="WhealVision API",
    description="Automated AI-powered detection and measurement of skin-prick allergy wheals",
    version="1.0.0",
)

# Configure CORS so Vercel frontend can call endpoints without restriction
custom_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST API endpoints
custom_app.include_router(measurements.router, prefix="/api/v1")
custom_app.get("/health")(health_check)


# ── 2. Gradio GPU Handler ──
@spaces.GPU(duration=120)
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


# ── 3. Build Gradio Interactive Web Interface ──
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


# ── 4. Launch Gradio Demo with custom_app mounted ──
# Hugging Face ZeroGPU requires demo.launch() to discover @spaces.GPU functions!
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            _app=custom_app,
            prevent_thread_lock=False,
        )
    except TypeError:
        # Fallback if launch() doesn't accept _app
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            prevent_thread_lock=False,
        )
