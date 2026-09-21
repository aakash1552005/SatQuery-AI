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

from src.contracts.query_contracts import (
    AnalysisResult,
    QueryIntent,
    RoutingDecision,
    TaskType,
)
from src.contracts.raster_contracts import (
    PairCompatibility,
    RasterMetadata,
    UploadResponse,
)
from src.execution.trace_engine import ExecutionTrace, TraceEngine
from src.gateway.compatibility_checker import CompatibilityChecker
from src.gateway.raster_inspector import RasterInspector
from src.router.agentic_router import AgenticRouter
from src.router.capability_registry import CapabilityRegistry, RuntimeMode

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
    version="0.2.0-day2",
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
registry = CapabilityRegistry(runtime_mode=RuntimeMode.DEMO_FALLBACK)
router = AgenticRouter(registry=registry)

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
    return {"status": "ok", "service": "satquery-ai", "version": "0.2.0-day2"}


@app.get("/api/status", response_model=SystemStatus)
async def system_status():
    return SystemStatus(
        status="running",
        version="0.2.0-day2",
        runtime_mode=registry.runtime_mode.value,
        capabilities={name: rec["status"] for name, rec in registry.get_all().items()},
    )


@app.get("/api/capabilities")
async def get_capabilities():
    """Retrieve complete live capability registry with engine notes and license status."""
    return {
        "runtime_mode": registry.runtime_mode.value,
        "capabilities": registry.get_all(),
    }


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
# Query & Agentic Routing (Day 2)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    file_ids: list[str] = []


class QueryResponse(BaseModel):
    decision: RoutingDecision
    trace: ExecutionTrace
    result: Optional[AnalysisResult] = None
    status: str


@app.post("/api/query", response_model=QueryResponse)
async def process_query(req: QueryRequest):
    """
    Process natural language user query against selected raster inputs.
    Emits sensor-aware routing decision, transparent pathway, and execution trace.
    """
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    # Resolve metadata objects for provided file_ids
    input_metadata: list[RasterMetadata] = []
    for fid in req.file_ids:
        if fid not in _uploaded_files:
            raise HTTPException(404, f"Referenced file not found: {fid}")
        input_metadata.append(_uploaded_files[fid]["metadata"])

    # Dynamic routing
    decision = router.route(req.query, input_metadata)

    # Build factual execution trace (Section 24)
    trace = TraceEngine.create_trace(
        task_type=decision.task_type.value,
        pathway=decision.pathway_label,
        runtime_mode=decision.runtime_mode,
        tools_selected=decision.tool_sequence,
        tools_executed=[te.model_dump() for te in decision.tool_executions],
    )

    trace.add_step(
        "Input validation",
        "completed" if decision.provided_inputs_count > 0 else "failed",
        f"{decision.provided_inputs_count} image(s) verified",
    )

    if input_metadata:
        trace.add_step(
            "GeoTIFF metadata extracted",
            "completed",
            f"Formats: {[m.format for m in input_metadata]}, Bands: {[m.band_count for m in input_metadata]}",
        )
        trace.add_step(
            "Sensor detected",
            "completed",
            f"Modalities: {[m.modality.value for m in input_metadata]} (detection_method: {[m.detection_method for m in input_metadata]})",
        )

    trace.add_step(
        "Query classified",
        "completed",
        f"Task: {decision.task_type.value} (classifier: rule_based, confidence_type: heuristic)",
    )

    trace.add_step(
        "Pathway selected",
        "completed" if decision.is_executable else "refused",
        decision.pathway_label,
    )

    if not decision.is_executable:
        trace.add_step(
            "Routing refusal",
            "failed",
            decision.refusal_reason or "Execution stopped per routing rules",
        )
        status = "refused"
    else:
        trace.add_step(
            "Tool execution sequence prepared",
            "completed",
            f"{len(decision.tool_sequence)} tool(s) scheduled",
        )
        status = "routed"

    decision.execution_trace_id = trace.trace_id

    # Construct honest AnalysisResult contract (Section 0 & Section 14)
    analysis_result = AnalysisResult(
        task=decision.task_type,
        mechanism=(
            "deterministic_sar_analysis"
            if decision.task_type == TaskType.SINGLE_IMAGE_VQA_SAR
            else (
                "deterministic_optical_spectral_analysis"
                if decision.task_type == TaskType.SINGLE_IMAGE_VQA_OPTICAL
                else None
            )
        ),
        model=None,
        answer=None,
        evidence=[],
        confidence={"type": "heuristic", "calibrated": False},
        limitations=[
            "Routing & readiness verified. Numerical execution engine scheduled in subsequent milestone."
        ],
        status="ROUTED_PENDING_EXECUTION" if decision.is_executable else "REFUSED",
    )

    return QueryResponse(
        decision=decision,
        trace=trace,
        result=analysis_result,
        status=status,
    )



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
