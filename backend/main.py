from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.v1 import measurements

app = FastAPI(title="Allergy Wheal Detection API", version="1.0.0")

# Add CORS middleware to allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(measurements.router, prefix="/api/v1")


@app.get("/")
def health_check():
    return {"status": "active", "version": "0.1.0"}

                