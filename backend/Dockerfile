# Production Dockerfile for Allergy Wheal Detection API
# Compatible with: Hugging Face Spaces (Port 7860), Google Cloud Run (Port 8080), Docker Compose (Port 8000)

FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr and writing bytecode
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies required for OpenCV, image processing, and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU build to keep Docker image lightweight and fast
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Meta Segment Anything Model
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# Install requirements
COPY requirements.txt .
# Filter out torch/torchvision/segment-anything from requirements.txt to prevent overwriting CPU builds
RUN grep -v -E "(torch|torchvision|segment-anything)" requirements.txt > requirements.deploy.txt \
    && pip install --no-cache-dir -r requirements.deploy.txt \
    && rm requirements.deploy.txt

# Copy backend source
COPY . .

# Download SAM ViT-B weights during image build so the container starts immediately without network delay
RUN python -m scripts.download_sam

# Expose default API port and Hugging Face default port
EXPOSE 8000
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Launch uvicorn dynamically binding to $PORT (handles Hugging Face, Cloud Run, Render, VPS)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
