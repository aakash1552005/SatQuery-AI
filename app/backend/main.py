"""
SatQuery AI -- FastAPI Backend
Section 25: Backend API for upload, metadata inspection, compatibility checking.
Day 1 scope: upload + inspect + validate.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.contracts.raster_contracts import (
    PairCompatibility,
    RasterMetadata,
    UploadResponse,
)
from src.gateway.compatibility_checker import CompatibilityChecker
from src.gateway.raster_inspector import RasterInspector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("satquery.api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SatQuery AI",
    description=(
        "An Interactive Vision-Language Assistant for Multimodal "
        "Remote Sensing Image Analysis through Text Queries. "
        "PS 26167 -- ISRO / Department of Space / SAC."
    ),
    version="0.1.0-day1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

inspector = RasterInspector()
compat_checker = CompatibilityChecker()

# In-memory store of uploaded file metadata (keyed by file_id)
_uploaded_files: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------

class SystemStatus(BaseModel):
    status: str
    version: str
    runtime_mode: str
    capabilities: dict


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "satquery-ai", "version": "0.1.0-day1"}


@app.get("/api/status", response_model=SystemStatus)
async def system_status():
    return SystemStatus(
        status="running",
        version="0.1.0-day1",
        runtime_mode="DEMO_FALLBACK",
        capabilities={
            "upload": "READY",
            "metadata_inspection": "READY",
            "compatibility_check": "READY",
            "single_image_vqa_optical": "NOT_IMPLEMENTED",
            "single_image_vqa_sar": "NOT_IMPLEMENTED",
            "single_image_grounding": "NOT_IMPLEMENTED",
            "temporal_change": "NOT_IMPLEMENTED",
            "optical_sar_fusion": "NOT_IMPLEMENTED",
            "geochat": "UNAVAILABLE",
            "changechat": "BLOCKED_LICENSE",
            "croma": "DISABLED",
            "sar_deterministic_tools": "NOT_IMPLEMENTED",
        },
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a raster file (GeoTIFF, TIFF, PNG, JPEG).
    Returns extracted metadata and validation status.
    """
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    # Save to disk
    file_id = str(uuid.uuid4())[:12]
    safe_name = file.filename.replace(" ", "_")
    dest = UPLOAD_DIR / f"{file_id}_{safe_name}"

    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {e}")
    finally:
        await file.close()

    # Inspect
    metadata = inspector.inspect(str(dest))
    metadata.file_id = file_id  # Override with our generated ID

    # Store reference
    _uploaded_files[file_id] = {
        "path": str(dest),
        "metadata": metadata,
    }

    # Determine status
    status = "accepted" if metadata.is_valid else "rejected"
    rejection = None
    if not metadata.is_valid:
        rejection = "; ".join(metadata.validation_errors)

    logger.info(
        "Upload %s: %s [%s] %dx%d bands=%d modality=%s",
        status, file.filename, metadata.format,
        metadata.width, metadata.height,
        metadata.band_count, metadata.modality.value,
    )

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        metadata=metadata,
        status=status,
        rejection_reason=rejection,
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@app.get("/api/files/{file_id}/metadata", response_model=RasterMetadata)
async def get_metadata(file_id: str):
    """Retrieve metadata for a previously uploaded file."""
    if file_id not in _uploaded_files:
        raise HTTPException(404, f"File not found: {file_id}")
    return _uploaded_files[file_id]["metadata"]


@app.get("/api/files")
async def list_files():
    """List all uploaded files and their metadata summaries."""
    result = []
    for fid, data in _uploaded_files.items():
        meta: RasterMetadata = data["metadata"]
        result.append({
            "file_id": fid,
            "filename": meta.filename,
            "modality": meta.modality.value,
            "dimensions": f"{meta.width}x{meta.height}",
            "bands": meta.band_count,
            "crs": meta.crs,
            "is_valid": meta.is_valid,
        })
    return result


# ---------------------------------------------------------------------------
# Compatibility Check
# ---------------------------------------------------------------------------

class CompatibilityRequest(BaseModel):
    file_id_a: str
    file_id_b: str


@app.post("/api/compatibility", response_model=PairCompatibility)
async def check_compatibility(req: CompatibilityRequest):
    """Check compatibility between two uploaded rasters."""
    if req.file_id_a not in _uploaded_files:
        raise HTTPException(404, f"File not found: {req.file_id_a}")
    if req.file_id_b not in _uploaded_files:
        raise HTTPException(404, f"File not found: {req.file_id_b}")

    meta_a = _uploaded_files[req.file_id_a]["metadata"]
    meta_b = _uploaded_files[req.file_id_b]["metadata"]

    result = compat_checker.check_pair(meta_a, meta_b)
    return result


# ---------------------------------------------------------------------------
# Serve frontend (static files)
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path("app/frontend")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True),
              name="frontend")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
