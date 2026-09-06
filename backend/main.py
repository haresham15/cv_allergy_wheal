import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import config
from routers.v1 import measurements

app = FastAPI(
    title="Allergy Wheal Detection API",
    description="Automated AI-powered detection and measurement of skin-prick allergy wheals",
    version="1.0.0",
)

# Configure CORS for flexible deployment
origins = config.CORS_ORIGINS if config.CORS_ORIGINS and config.CORS_ORIGINS != ["*"] else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(measurements.router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "service": "Allergy Wheal Detection API",
        "status": "active",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    sam_exists = os.path.exists(config.SAM_CHECKPOINT_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "healthy" if sam_exists else "degraded",
        "version": "1.0.0",
        "device": device,
        "sam_model_available": sam_exists,
        "sam_model_path": config.SAM_CHECKPOINT_PATH,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT, reload=False)

                