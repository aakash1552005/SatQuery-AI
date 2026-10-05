"""
SatQuery AI -- Day 5 Automated Verification Suite
Tests for:
1. DeterministicFusionEngine (Optical-SAR cross-modal fusion, agreement matrix, ground km²)
2. DeterministicChangeEngine (Bi-temporal change detection, registration quality gate, L1/L2 declaration)
3. End-to-End API Integration via FastAPI /api/query for multimodal and temporal requests
4. CapabilityRegistry status verification (READY on CPU without GPU dependencies)
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from src.analysis.change_engine import DeterministicChangeEngine, RegistrationQualityGate
from src.analysis.fusion_engine import DeterministicFusionEngine
from src.contracts.query_contracts import TaskType, ToolExecutionStatus
from src.router.capability_registry import CapabilityRegistry, CapabilityStatus

client = TestClient(app)
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_optical_sar_fusion_engine_direct():
    """Verify direct mathematical execution of the Optical-SAR Fusion Engine."""
    opt_path = SAMPLES_DIR / "optical_sar" / "pair_optical.tif"
    sar_path = SAMPLES_DIR / "optical_sar" / "pair_sar.tif"

    assert opt_path.exists(), f"Sample missing: {opt_path}"
    assert sar_path.exists(), f"Sample missing: {sar_path}"

    engine = DeterministicFusionEngine()
    result, executed_tools = engine.analyze_pair(
        optical_path=opt_path,
        sar_path=sar_path,
        query="Identify flood extent beneath clouds and compute square kilometers",
    )

    assert result.status == "EXECUTED"
    assert result.mechanism == "deterministic_optical_sar_cross_modal_fusion"
    assert result.model is None
    assert "km²" in result.answer
    assert "Verified High-Confidence Water" in result.answer
    assert "Cloud-Piercing / Radar-Exclusive Water" in result.answer

    # Check decomposed confidence
    assert result.confidence["evidence_confidence"] in ("HIGH", "MODERATE")
    assert result.confidence["measurement_quality"] == "DETERMINISTIC_GIS_COMPUTED"
    assert result.confidence["calibrated"] is False

    # Check executed toolchain
    assert "OpticalEvidenceExtractor (NDWI)" in executed_tools
    assert "SARBackscatterExtractor (Lee+Otsu)" in executed_tools
    assert "CrossModalFusionEngine" in executed_tools
    assert "AgreementTierClassifier" in executed_tools


def test_bitemporal_change_engine_direct():
    """Verify direct mathematical execution of the Bi-Temporal Change Engine."""
    t1_path = SAMPLES_DIR / "temporal" / "temporal_t1_2024_jan.tif"
    t2_path = SAMPLES_DIR / "temporal" / "temporal_t2_2025_jan.tif"

    assert t1_path.exists(), f"Sample missing: {t1_path}"
    assert t2_path.exists(), f"Sample missing: {t2_path}"

    engine = DeterministicChangeEngine()
    result, executed_tools = engine.analyze_pair(
        t1_path=t1_path,
        t2_path=t2_path,
        query="What changed between the two dates?",
    )

    assert result.status == "EXECUTED"
    assert result.mechanism == "deterministic_bitemporal_change_engine"
    assert result.model is None
    assert "km²" in result.answer
    assert "PHYSICAL CHANGE (L1)" in result.answer
    assert "Registration Gate: PASS" in result.answer

    # Check mandatory L1/L2 declaration per Section 16.3
    assert result.confidence["level"] == "PHYSICAL_CHANGE_L1"
    assert result.confidence["semantic_interpretation"] == "NOT_AVAILABLE"

    # Check executed toolchain
    assert "RegistrationQualityGate (AROSICS)" in executed_tools
    assert "TemporalConfoundCheck" in executed_tools
    assert "DeterministicChangeEngine" in executed_tools
    assert "L1L2DeclarationGate" in executed_tools


def test_api_optical_sar_fusion_query():
    """Verify live API query processing for Optical + SAR multimodal pair."""
    opt_path = SAMPLES_DIR / "optical_sar" / "pair_optical.tif"
    sar_path = SAMPLES_DIR / "optical_sar" / "pair_sar.tif"

    with open(opt_path, "rb") as f_opt:
        resp_a = client.post("/api/upload", files={"file": ("fuse_opt.tif", f_opt, "image/tiff")})
    assert resp_a.status_code == 200
    fid_opt = resp_a.json()["file_id"]

    with open(sar_path, "rb") as f_sar:
        resp_b = client.post("/api/upload", files={"file": ("fuse_sar.tif", f_sar, "image/tiff")})
    assert resp_b.status_code == 200
    fid_sar = resp_b.json()["file_id"]

    q_resp = client.post(
        "/api/query",
        json={
            "query": "Perform optical-sar fusion to delineate floodwater under monsoon clouds",
            "file_ids": [fid_opt, fid_sar],
        },
    )
    assert q_resp.status_code == 200
    data = q_resp.json()

    assert data["status"] == "routed"
    assert data["decision"]["task_type"] == TaskType.OPTICAL_SAR_ANALYSIS
    assert data["decision"]["is_executable"] is True

    result = data["result"]
    assert result["status"] == "EXECUTED"
    assert result["mechanism"] == "deterministic_optical_sar_cross_modal_fusion"
    assert "km²" in result["answer"]

    # Verify tool execution status
    tool_execs = {t["tool_name"]: t["status"] for t in data["decision"]["tool_executions"]}
    assert tool_execs.get("CrossModalFusionEngine") == ToolExecutionStatus.EXECUTED.value


def test_api_temporal_change_query():
    """Verify live API query processing for Bi-Temporal observation pair."""
    t1_path = SAMPLES_DIR / "temporal" / "temporal_t1_2024_jan.tif"
    t2_path = SAMPLES_DIR / "temporal" / "temporal_t2_2025_jan.tif"

    with open(t1_path, "rb") as f_t1:
        resp_a = client.post("/api/upload", files={"file": ("t1.tif", f_t1, "image/tiff")})
    assert resp_a.status_code == 200
    fid_t1 = resp_a.json()["file_id"]

    with open(t2_path, "rb") as f_t2:
        resp_b = client.post("/api/upload", files={"file": ("t2.tif", f_t2, "image/tiff")})
    assert resp_b.status_code == 200
    fid_t2 = resp_b.json()["file_id"]

    q_resp = client.post(
        "/api/query",
        json={
            "query": "Quantify temporal change and calculate surface area dynamics",
            "file_ids": [fid_t1, fid_t2],
        },
    )
    assert q_resp.status_code == 200
    data = q_resp.json()

    assert data["status"] == "routed"
    assert data["decision"]["task_type"] == TaskType.TEMPORAL_CHANGE
    assert data["decision"]["is_executable"] is True

    result = data["result"]
    assert result["status"] == "EXECUTED"
    assert result["mechanism"] == "deterministic_bitemporal_change_engine"
    assert result["confidence"]["level"] == "PHYSICAL_CHANGE_L1"


def test_capability_registry_updates():
    """Verify CapabilityRegistry accurately reflects READY status for fusion and change engines."""
    registry = CapabilityRegistry()
    capabilities = registry.get_all()

    assert capabilities["temporal_change"]["status"] == CapabilityStatus.READY.value
    assert capabilities["temporal_change"]["execution_readiness"] == "READY"

    assert capabilities["optical_sar_fusion"]["status"] == CapabilityStatus.READY.value
    assert capabilities["optical_sar_fusion"]["execution_readiness"] == "READY"
