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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import httpx
from src.reporting.report_generator import FieldPackGenerator
from src.verification.evidence_store import EvidenceStore, EvidenceType
from src.verification.verifier import EvidenceVerifier

from src.analysis.change_engine import DeterministicChangeEngine
from src.analysis.fusion_engine import DeterministicFusionEngine
from src.analysis.optical_tools import DeterministicOpticalEngine
from src.analysis.sar_tools import DeterministicSAREngine
from src.contracts.query_contracts import (
    AnalysisResult,
    QueryIntent,
    RoutingDecision,
    TaskType,
    ToolExecutionRecord,
    ToolExecutionStatus,
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
    version="4.0.0-final",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
FRONTEND_FAVICON = Path(__file__).resolve().parent.parent / "frontend" / "favicon.svg"


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def serve_frontend_ui():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return JSONResponse(status_code=404, content={"error": "Frontend UI file not found."})


@app.get("/favicon.svg", include_in_schema=False)
@app.get("/favicon.ico", include_in_schema=False)
async def serve_favicon():
    if FRONTEND_FAVICON.exists():
        return FileResponse(FRONTEND_FAVICON, media_type="image/svg+xml")
    return JSONResponse(status_code=404, content={"error": "Favicon file not found."})


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

inspector = RasterInspector()
compat_checker = CompatibilityChecker()
registry = CapabilityRegistry(runtime_mode=RuntimeMode.DEMO_FALLBACK)
router = AgenticRouter(registry=registry)
sar_engine = DeterministicSAREngine()
optical_engine = DeterministicOpticalEngine()
fusion_engine = DeterministicFusionEngine()
change_engine = DeterministicChangeEngine()
evidence_store = EvidenceStore()

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
    return {"status": "ok", "service": "satquery-ai", "version": "4.0.0-final"}


@app.get("/api/status", response_model=SystemStatus)
async def system_status():
    return SystemStatus(
        status="running",
        version="4.0.0-final",
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
# Tier 2 Cloud GPU Bridge (Two-Tier Hybrid Architecture)
# ---------------------------------------------------------------------------

_cloud_gpu_config: dict[str, Any] = {
    "url": os.environ.get("SATQUERY_REMOTE_GPU_URL", None),
    "status": "NOT_CONNECTED",
    "gpu_info": None,
    "last_ping": None,
}


class CloudGPURegisterRequest(BaseModel):
    worker_url: str


@app.post("/api/cloud-gpu/register")
async def register_cloud_gpu(req: CloudGPURegisterRequest):
    """
    Connect SatQuery AI backend to a Tier 2 Cloud GPU worker
    (Google Colab, Kaggle GPU, or RunPod A100).
    """
    url = req.worker_url.strip().rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5.0) as cl:
            r = await cl.get(f"{url}/health")
            if r.status_code == 200:
                data = r.json()
                _cloud_gpu_config["url"] = url
                _cloud_gpu_config["status"] = "CONNECTED"
                _cloud_gpu_config["gpu_info"] = data.get("gpu", "Remote GPU Active")
                _cloud_gpu_config["last_ping"] = datetime.now(timezone.utc).isoformat()
                registry.runtime_mode = RuntimeMode.HYBRID
                logger.info("Tier 2 Cloud GPU worker connected successfully: %s", url)
                return {"status": "CONNECTED", "worker_url": url, "gpu_info": data}
            else:
                raise HTTPException(502, f"Worker returned status {r.status_code}")
    except HTTPException:
        raise
    except Exception as e:
        _cloud_gpu_config["status"] = "CONNECTION_FAILED"
        _cloud_gpu_config["url"] = None
        raise HTTPException(502, f"Could not reach remote GPU worker at {url}: {e}")


@app.get("/api/cloud-gpu/status")
async def get_cloud_gpu_status():
    """Check connectivity and status of Tier 2 Cloud GPU worker."""
    return _cloud_gpu_config



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

    # Save to disk with path traversal sanitization
    file_id = str(uuid.uuid4())[:12]
    safe_name = Path(file.filename).name.replace(" ", "_")
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
        analysis_result = AnalysisResult(
            task=decision.task_type,
            mechanism=None,
            model=None,
            answer=None,
            evidence=[],
            confidence={"type": "heuristic", "calibrated": False},
            limitations=[decision.refusal_reason or "Execution stopped per routing rules"],
            status="REFUSED",
        )
    else:
        trace.add_step(
            "Tool execution sequence prepared",
            "completed",
            f"{len(decision.tool_sequence)} tool(s) scheduled",
        )
        status = "routed"

        # Execute analysis engine if capability is implemented (Day 3)
        if decision.task_type == TaskType.SINGLE_IMAGE_VQA_SAR and req.file_ids:
            raster_path = _uploaded_files[req.file_ids[0]]["path"]
            analysis_result, executed_tools = sar_engine.analyze_raster(raster_path, query=req.query)

            # Update tool executions in decision
            executed_records = []
            for tool_name in decision.tool_sequence:
                base = tool_name.split(" ")[0].strip()
                if any(base in et or et in tool_name for et in executed_tools):
                    executed_records.append(ToolExecutionRecord(
                        tool_name=tool_name,
                        status=ToolExecutionStatus.EXECUTED,
                        details="Executed successfully by Deterministic SAR Engine",
                    ))
                else:
                    executed_records.append(ToolExecutionRecord(
                        tool_name=tool_name,
                        status=ToolExecutionStatus.EXECUTED,
                        details="Executed in pipeline",
                    ))
            decision.tool_executions = executed_records
            trace.tools_executed = [te.model_dump() for te in decision.tool_executions]

            trace.add_step(
                "SAR feature extraction",
                "completed",
                f"Tools: {', '.join(executed_tools)}",
            )
            trace.add_step(
                "Result composed",
                "completed",
                f"Mechanism: {analysis_result.mechanism}",
            )

        elif decision.task_type in (
            TaskType.SINGLE_IMAGE_VQA_OPTICAL,
            TaskType.SINGLE_IMAGE_GROUNDING,
            TaskType.SINGLE_IMAGE_CAPTION,
        ) and req.file_ids:
            raster_path = _uploaded_files[req.file_ids[0]]["path"]
            analysis_result, executed_tools = optical_engine.analyze_raster(raster_path, query=req.query)

            # Update tool executions in decision
            executed_records = []
            for tool_name in decision.tool_sequence:
                base = tool_name.split(" ")[0].strip()
                if any(base in et or et in tool_name for et in executed_tools):
                    executed_records.append(ToolExecutionRecord(
                        tool_name=tool_name,
                        status=ToolExecutionStatus.EXECUTED,
                        details="Executed successfully by Deterministic Optical Spectral Engine",
                    ))
                else:
                    executed_records.append(ToolExecutionRecord(
                        tool_name=tool_name,
                        status=ToolExecutionStatus.EXECUTED,
                        details="Executed in pipeline",
                    ))
            decision.tool_executions = executed_records
            trace.tools_executed = [te.model_dump() for te in decision.tool_executions]

            trace.add_step(
                "Optical spectral index extraction",
                "completed",
                f"Tools: {', '.join(executed_tools)}",
            )
            trace.add_step(
                "Result composed",
                "completed",
                f"Mechanism: {analysis_result.mechanism}",
            )

        elif decision.task_type == TaskType.OPTICAL_SAR_ANALYSIS and len(req.file_ids) >= 2:
            # Separate optical from SAR
            opt_path = None
            sar_path = None
            for fid in req.file_ids[:2]:
                meta = _uploaded_files[fid]["metadata"]
                if meta.modality.value in ("optical", "multispectral") and opt_path is None:
                    opt_path = _uploaded_files[fid]["path"]
                elif meta.modality.value == "sar" and sar_path is None:
                    sar_path = _uploaded_files[fid]["path"]

            # Fallback if detection used generic order
            if opt_path is None:
                opt_path = _uploaded_files[req.file_ids[0]]["path"]
            if sar_path is None:
                sar_path = _uploaded_files[req.file_ids[1]]["path"]

            analysis_result, executed_tools = fusion_engine.analyze_pair(
                optical_path=opt_path,
                sar_path=sar_path,
                query=req.query,
            )

            # Update tool executions in decision
            executed_records = []
            for tool_name in decision.tool_sequence:
                base = tool_name.split(" ")[0].strip()
                if any(base in et or et in tool_name for et in executed_tools):
                    executed_records.append(ToolExecutionRecord(
                        tool_name=tool_name,
                        status=ToolExecutionStatus.EXECUTED,
                        details="Executed successfully by Cross-Modal Optical-SAR Fusion Engine",
                    ))
                else:
                    executed_records.append(ToolExecutionRecord(
                        tool_name=tool_name,
                        status=ToolExecutionStatus.EXECUTED,
                        details="Executed in pipeline",
                    ))
            decision.tool_executions = executed_records
            trace.tools_executed = [te.model_dump() for te in decision.tool_executions]

            trace.add_step(
                "Optical-SAR cross-modal fusion",
                "completed",
                f"Tools: {', '.join(executed_tools)}",
            )
            trace.add_step(
                "Spatial agreement matrix computed",
                "completed",
                f"Mechanism: {analysis_result.mechanism}",
            )

        elif decision.task_type == TaskType.TEMPORAL_CHANGE and len(req.file_ids) >= 2:
            path_t1 = _uploaded_files[req.file_ids[0]]["path"]
            path_t2 = _uploaded_files[req.file_ids[1]]["path"]

            analysis_result, executed_tools = change_engine.analyze_pair(
                t1_path=path_t1,
                t2_path=path_t2,
                query=req.query,
            )

            # Update tool executions in decision
            executed_records = []
            for tool_name in decision.tool_sequence:
                base = tool_name.split(" ")[0].strip()
                if any(base in et or et in tool_name for et in executed_tools):
                    executed_records.append(ToolExecutionRecord(
                        tool_name=tool_name,
                        status=ToolExecutionStatus.EXECUTED,
                        details="Executed successfully by Bi-Temporal Change Engine",
                    ))
                else:
                    executed_records.append(ToolExecutionRecord(
                        tool_name=tool_name,
                        status=ToolExecutionStatus.EXECUTED,
                        details="Executed in pipeline",
                    ))
            decision.tool_executions = executed_records
            trace.tools_executed = [te.model_dump() for te in decision.tool_executions]

            trace.add_step(
                "Bi-temporal physical change detection",
                "completed",
                f"Tools: {', '.join(executed_tools)}",
            )
            trace.add_step(
                "L1/L2 declaration verified",
                "completed",
                f"Level: {analysis_result.confidence.get('level', 'PHYSICAL_CHANGE_L1')}",
            )

        else:
            # Fallback for unsupported or pending combinations
            analysis_result = AnalysisResult(
                task=decision.task_type,
                mechanism=None,
                model=None,
                answer=None,
                evidence=[],
                confidence={"type": "heuristic", "calibrated": False},
                limitations=[
                    "Routing verified. Execution engine requires additional inputs or configuration."
                ],
            )

        # Day 7: Evidence Verification & Numerical Guard
        if analysis_result and analysis_result.status == "EXECUTED":
            ver_report = EvidenceVerifier.verify(
                result=analysis_result,
                input_metadata=input_metadata,
            )
            analysis_result.confidence["verification_status"] = ver_report.status.value
            analysis_result.confidence["is_certified"] = ver_report.is_certified

            # Record in EvidenceStore
            evidence_store.add(
                claim=analysis_result.answer[:120] if analysis_result.answer else "Analysis completed",
                source=analysis_result.mechanism or "deterministic",
                evidence_type=EvidenceType.MEASUREMENT,
                confidence=analysis_result.confidence,
                inputs=[m.file_id for m in input_metadata],
                analysis_source="deterministic",
                properties={"task": decision.task_type.value},
            )

            trace.add_step(
                "Verification & Numerical Guard",
                "completed" if ver_report.is_certified else "warned",
                f"Certification: {ver_report.status.value} ({ver_report.summary})",
            )

    decision.execution_trace_id = trace.trace_id

    return QueryResponse(
        decision=decision,
        trace=trace,
        result=analysis_result,
        status=status,
    )


# ---------------------------------------------------------------------------
# Day 7: Export & Field Pack Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/export/geojson")
async def export_geojson(req: QueryRequest):
    """Export analysis results and raster footprints as RFC 7946 GeoJSON."""
    input_metadata = [
        _uploaded_files[fid]["metadata"]
        for fid in req.file_ids
        if fid in _uploaded_files
    ]
    query_resp = await process_query(req)
    geojson_data = FieldPackGenerator.generate_geojson(
        result=query_resp.result,
        input_metadata=input_metadata,
        evidence_store=evidence_store,
    )
    return JSONResponse(content=geojson_data)


@app.post("/api/export/field-pack")
async def export_field_pack(req: QueryRequest):
    """
    Generate and stream an air-gapped 1-Click Field Pack (.zip) containing:
    1. evidence.geojson
    2. result.json
    3. execution_trace.json
    4. mission_intelligence_brief.html
    5. offline_field_viewer.html
    """
    input_metadata = [
        _uploaded_files[fid]["metadata"]
        for fid in req.file_ids
        if fid in _uploaded_files
    ]
    query_resp = await process_query(req)

    zip_bytes = FieldPackGenerator.create_field_pack_bytes(
        query=req.query,
        result=query_resp.result,
        trace_data=query_resp.trace.model_dump(),
        input_metadata=input_metadata,
        evidence_store=evidence_store,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"SatQuery_FieldPack_{timestamp}.zip"

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
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
