"""API v1 — /analyze endpoint.

Accepts an image upload + optional allergen grid JSON.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import json

from core import config
from services.vision_pipeline import process_image

router = APIRouter()


@router.post("/analyze")
async def analyze_skin_test(
    file: UploadFile = File(...),
    allergen_grid: Optional[str] = Form(None),
    body_location: Optional[str] = Form(None),
):
    """Analyse an allergy skin-prick test image.

    Parameters
    ----------
    file : UploadFile
        JPEG or PNG photograph of the test site (with ArUco marker).
    allergen_grid : str (JSON), optional
        JSON-encoded dict mapping grid positions to allergen names.
        Example: '{"A1": "Peanut", "A2": "Dust Mite", "B1": "Cat Dander"}'
    body_location : str, optional
        Anatomical location ("forearm" or "back") to refine calibration when no marker is present.
    """

    # ── Validate file ──
    is_valid_type = (
        file.content_type in config.ALLOWED_CONTENT_TYPES
        or (file.content_type and file.content_type.startswith("image/"))
        or (
            file.filename
            and file.filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif")
            )
        )
    )
    if not is_valid_type:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported formats: JPEG, PNG, WebP, BMP, TIFF, HEIC.",
        )

    contents = await file.read()
    if len(contents) > config.MAX_UPLOAD_SIZE:
        max_mb = config.MAX_UPLOAD_SIZE // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File too large (max {max_mb} MB).")

    # ── Parse allergen grid ──
    grid_dict = None
    if allergen_grid:
        try:
            grid_dict = json.loads(allergen_grid)
            if not isinstance(grid_dict, dict):
                raise ValueError()
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="allergen_grid must be a valid JSON object, e.g. "
                       '\'{"A1": "Peanut", "A2": "Dust Mite"}\'',
            )

    # ── Process ──
    try:
        results = process_image(contents, allergen_grid=grid_dict, body_location=body_location)
        return JSONResponse(content=results)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal processing error")