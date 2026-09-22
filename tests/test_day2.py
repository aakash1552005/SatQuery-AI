"""
SatQuery AI -- Day 2 Automated Acceptance Tests
Tests for:
1. QueryParser (Natural language intent classification)
2. CapabilityRegistry (Status, license notes, fallback mapping)
3. AgenticRouter:
   - Optical vs SAR pathway separation (Section 9)
   - Missing temporal observation refusal (Section 15 & DEMO 6)
   - Optical-SAR multimodal fusion routing
   - Zero hallucination guarantees
4. ExecutionTrace engine (Factual step logging)
5. Backend API endpoints (/api/capabilities, /api/query)
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts.query_contracts import PathwayType, TaskType
from src.contracts.raster_contracts import RasterMetadata, SensorModality
from src.execution.trace_engine import TraceEngine
from src.gateway.raster_inspector import RasterInspector
from src.router.agentic_router import AgenticRouter
from src.router.capability_registry import CapabilityRegistry, CapabilityStatus, RuntimeMode
from src.router.query_parser import QueryParser
from app.backend.main import app

client = TestClient(app)
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
inspector = RasterInspector()


def test_query_parser_classification():
    """Verify natural language query parsing across all target task types."""
    parser = QueryParser()

    # Temporal change query
    t_intent = parser.parse("What changed between the two observations in 2024 and 2025?")
    assert t_intent.task_type == TaskType.TEMPORAL_CHANGE
    assert t_intent.requires_temporal_pair is True
    assert t_intent.requires_spatial_evidence is True

    # Grounding query
    g_intent = parser.parse("Where is the largest water body? Locate and highlight with bounding box.")
    assert g_intent.task_type == TaskType.SINGLE_IMAGE_GROUNDING
    assert g_intent.requires_spatial_evidence is True
    assert "water body" in g_intent.target_entities

    # Optical-SAR fusion query
    f_intent = parser.parse("Fuse optical and SAR observations to detect surface water.")
    assert f_intent.task_type == TaskType.OPTICAL_SAR_ANALYSIS
    assert f_intent.requires_optical_sar is True

    # Explicit SAR query
    s_intent = parser.parse("Analyze the radar backscatter and VV/VH polarization ratio.")
    assert s_intent.task_type == TaskType.SINGLE_IMAGE_VQA_SAR

    # General optical query
    o_intent = parser.parse("What type of land cover dominates this region?")
    assert o_intent.task_type == TaskType.SINGLE_IMAGE_VQA_OPTICAL


def test_capability_registry():
    """Verify capability registry state reporting and separated readiness indicators."""
    reg = CapabilityRegistry(runtime_mode=RuntimeMode.DEMO_FALLBACK)
    all_caps = reg.get_all()

    assert "single_image_vqa_sar" in all_caps
    assert all_caps["single_image_vqa_sar"]["routing_readiness"] == "READY"
    assert all_caps["single_image_vqa_sar"]["execution_readiness"] in ("NOT_IMPLEMENTED", "READY")
    assert all_caps["single_image_vqa_sar"]["status"] in ("NOT_IMPLEMENTED", "READY")
    assert "geochat" in all_caps
    assert all_caps["geochat"]["status"] == "UNAVAILABLE"
    assert "changechat" in all_caps
    assert all_caps["changechat"]["status"] == "BLOCKED_LICENSE"


def test_sar_pathway_separation():
    """
    Section 9: SAR imagery MUST be routed to SAR deterministic tools,
    NEVER silently converted and fed to an optical VLM.
    """
    router = AgenticRouter()
    sar_path = SAMPLES_DIR / "sar" / "synthetic_sar_vv_vh.tif"
    sar_meta = inspector.inspect(str(sar_path))
    assert sar_meta.modality == SensorModality.SAR

    # User asks a generic question on a SAR image
    decision = router.route("What is visible in this scene?", [sar_meta])

    assert decision.task_type == TaskType.SINGLE_IMAGE_VQA_SAR
    assert decision.pathway == PathwayType.SAR_DETERMINISTIC_TOOLS
    assert "SAR Deterministic" in decision.pathway_label
    assert decision.is_executable is True
    assert "SARBackscatterAnalysis" in decision.tool_sequence[1]
    assert len(decision.tool_executions) > 0
    assert decision.tool_executions[0].status.value == "EXECUTED"
    assert decision.tool_executions[1].status.value == "NOT_IMPLEMENTED"


def test_missing_input_refusal_demo6():
    """
    DEMO 6 & Section 15: If user asks for temporal change with only 1 image,
    refuse execution with technical reason. NEVER hallucinate a second image.
    """
    router = AgenticRouter()
    opt_path = SAMPLES_DIR / "optical" / "synthetic_optical_rgb.tif"
    opt_meta = inspector.inspect(str(opt_path))

    decision = router.route("What changed in this area?", [opt_meta])

    assert decision.is_executable is False
    assert decision.pathway == PathwayType.REFUSAL
    assert "Only one acquisition was supplied" in decision.refusal_reason
    assert "Required: Two temporally distinct observations" in decision.refusal_reason
    assert len(decision.tool_sequence) == 0


def test_multimodal_fusion_routing():
    """Verify optical-SAR fusion routing when both modalities are supplied vs missing."""
    router = AgenticRouter()
    opt_meta = inspector.inspect(str(SAMPLES_DIR / "optical" / "synthetic_optical_rgb.tif"))
    sar_meta = inspector.inspect(str(SAMPLES_DIR / "sar" / "synthetic_sar_vv_vh.tif"))

    # Both supplied
    decision_both = router.route("Fuse both sensors to map surface water", [opt_meta, sar_meta])
    assert decision_both.is_executable is True
    assert decision_both.task_type == TaskType.OPTICAL_SAR_ANALYSIS
    assert decision_both.pathway == PathwayType.OPTICAL_SAR_FUSION

    # Only optical supplied for fusion query -> Refusal
    decision_missing = router.route("Fuse both sensors to map surface water", [opt_meta])
    assert decision_missing.is_executable is False
    assert decision_missing.pathway == PathwayType.REFUSAL
    assert "Missing Modality" in decision_missing.pathway_label


def test_execution_trace_engine():
    """Verify trace engine step logging and text formatting."""
    trace = TraceEngine.create_trace(
        task_type="single_image_vqa_sar",
        pathway="SAR Deterministic Feature Tools",
        runtime_mode="DEMO_FALLBACK",
        tools_selected=["RasterInspector", "SARBackscatterAnalysis"],
    )
    trace.add_step("Input validation", "completed", "1 file verified", 2.5)
    trace.add_step("Sensor detected", "completed", "SAR (VV+VH)", 1.2)

    text = trace.format_text()
    assert "EXECUTION TRACE:" in text
    assert "✓ Input validation" in text
    assert "✓ Sensor detected" in text
    assert "SARBackscatterAnalysis" in text


def test_api_capabilities_endpoint():
    """Verify GET /api/capabilities returns structured registry."""
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "runtime_mode" in data
    assert "capabilities" in data
    assert "single_image_vqa_sar" in data["capabilities"]


def test_api_query_flow():
    """Verify POST /api/query with uploads and routing."""
    # 1. Upload SAR image
    sar_path = SAMPLES_DIR / "sar" / "synthetic_sar_vv_vh.tif"
    with open(sar_path, "rb") as f:
        upload_resp = client.post("/api/upload", files={"file": ("sar_test.tif", f, "image/tiff")})
    assert upload_resp.status_code == 200
    file_id = upload_resp.json()["file_id"]

    # 2. Query with SAR file
    q_resp = client.post(
        "/api/query",
        json={"query": "Analyze radar roughness and backscatter", "file_ids": [file_id]},
    )
    assert q_resp.status_code == 200
    res_data = q_resp.json()
    assert res_data["status"] == "routed"
    assert res_data["decision"]["pathway"] == "sar_deterministic_tools"
    assert "result" in res_data
    assert res_data["result"]["mechanism"] == "deterministic_sar_analysis"
    assert len(res_data["trace"]["steps"]) >= 4

    # 3. Query with temporal request but only 1 file -> Refusal
    refusal_resp = client.post(
        "/api/query",
        json={"query": "What changed between 2024 and 2025?", "file_ids": [file_id]},
    )
    assert refusal_resp.status_code == 200
    ref_data = refusal_resp.json()
    assert ref_data["status"] == "refused"
    assert ref_data["decision"]["is_executable"] is False
    assert "Only one acquisition was supplied" in ref_data["decision"]["refusal_reason"]
