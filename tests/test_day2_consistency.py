"""
SatQuery AI — Day 1 & Day 2 Consistency & Status Integrity Test Suite
Section 0, Section 9, Section 10, Section 23 of Master Specification v4.

Verifies:
1. test_optical_mechanism_matches_actual_execution
2. test_sar_not_implemented_status
3. test_sar_scheduled_vs_executed
4. test_geochat_preflight_does_not_claim_real_inference
5. test_capability_registry_matches_actual_execution_state
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from src.router.capability_registry import CapabilityRegistry, CapabilityStatus, RuntimeMode
from src.contracts.query_contracts import TaskType, PathwayType, ToolExecutionStatus

client = TestClient(app)
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_optical_mechanism_matches_actual_execution():
    """
    Verify optical query execution path:
    Routing is READY; tool execution marks RasterInspector as EXECUTED,
    downstream analysis as NOT_IMPLEMENTED, and result as ROUTED_PENDING_EXECUTION.
    Does not falsely claim full VQA or land-cover classification is executed.
    """
    opt_path = SAMPLES_DIR / "optical" / "synthetic_optical_rgb.tif"
    with open(opt_path, "rb") as f:
        up_resp = client.post("/api/upload", files={"file": ("opt_test.tif", f, "image/tiff")})
    assert up_resp.status_code == 200
    file_id = up_resp.json()["file_id"]

    q_resp = client.post(
        "/api/query",
        json={"query": "What land cover dominates this optical scene?", "file_ids": [file_id]},
    )
    assert q_resp.status_code == 200
    data = q_resp.json()

    # Routing verified
    assert data["decision"]["task_type"] == TaskType.SINGLE_IMAGE_VQA_OPTICAL
    assert data["decision"]["pathway"] == PathwayType.OPTICAL_DETERMINISTIC
    assert data["decision"]["is_executable"] is True

    # Truthful mechanism contract verified
    res = data["result"]
    assert res["mechanism"] == "deterministic_optical_spectral_analysis"
    assert res["model"] is None  # VLM Not Used
    assert res["status"] == "ROUTED_PENDING_EXECUTION"
    assert res["answer"] is None  # Factual: no fake generated answer before Day 3 execution engine


def test_sar_not_implemented_status():
    """
    Verify SAR query routing to deterministic pathway, but honestly declaring
    that concrete analysis tools remain NOT_IMPLEMENTED until Day 3.
    """
    sar_path = SAMPLES_DIR / "sar" / "synthetic_sar_vv_vh.tif"
    with open(sar_path, "rb") as f:
        up_resp = client.post("/api/upload", files={"file": ("sar_test.tif", f, "image/tiff")})
    assert up_resp.status_code == 200
    file_id = up_resp.json()["file_id"]

    q_resp = client.post(
        "/api/query",
        json={"query": "Analyze radar roughness and backscatter", "file_ids": [file_id]},
    )
    assert q_resp.status_code == 200
    data = q_resp.json()

    assert data["decision"]["task_type"] == TaskType.SINGLE_IMAGE_VQA_SAR
    assert data["decision"]["pathway"] == PathwayType.SAR_DETERMINISTIC_TOOLS
    assert data["result"]["mechanism"] == "deterministic_sar_analysis"
    assert data["result"]["status"] == "ROUTED_PENDING_EXECUTION"


def test_sar_scheduled_vs_executed():
    """
    Verify scheduled tools vs executed tools separation:
    RasterInspector is EXECUTED.
    SARBackscatterAnalysis, LeeSpeckleFilter, PolarizationRatioEstimator,
    SARStructuredResponseComposer are NOT_IMPLEMENTED.
    """
    sar_path = SAMPLES_DIR / "sar" / "synthetic_sar_vv_vh.tif"
    with open(sar_path, "rb") as f:
        up_resp = client.post("/api/upload", files={"file": ("sar_test2.tif", f, "image/tiff")})
    assert up_resp.status_code == 200
    file_id = up_resp.json()["file_id"]

    q_resp = client.post(
        "/api/query",
        json={"query": "Measure VV and VH backscatter in dB", "file_ids": [file_id]},
    )
    assert q_resp.status_code == 200
    data = q_resp.json()

    tool_execs = {t["tool_name"]: t["status"] for t in data["decision"]["tool_executions"]}
    assert tool_execs["RasterInspector"] == ToolExecutionStatus.EXECUTED.value

    # Concrete SAR tools must be NOT_IMPLEMENTED
    sar_analysis_tools = [k for k in tool_execs if k != "RasterInspector"]
    assert len(sar_analysis_tools) >= 3
    for tool_name in sar_analysis_tools:
        assert tool_execs[tool_name] == ToolExecutionStatus.NOT_IMPLEMENTED.value


def test_geochat_preflight_does_not_claim_real_inference():
    """
    Verify GeoChat preflight matrix distinguishes environment preflight from real model inference.
    Must never report real inference as EXECUTED when only a stub or no GPU exists.
    """
    registry = CapabilityRegistry(runtime_mode=RuntimeMode.DEMO_FALLBACK)
    preflight = registry.get_geochat_preflight_status()

    assert preflight["repository"] in ("ABSENT", "PRESENT", "UNKNOWN")
    assert preflight["CUDA"] == "UNAVAILABLE"
    assert preflight["GPU_VRAM"] == "INSUFFICIENT"
    assert preflight["environment_preflight"] in ("PASSED", "FAILED", "NOT_EXECUTED")
    assert preflight["real_model_inference"] == "NOT_EXECUTED"
    assert preflight["final_capability"] == "UNAVAILABLE"


def test_capability_registry_matches_actual_execution_state():
    """
    Verify capability registry truthfully separates routing_readiness from execution_readiness.
    Capabilities whose analysis engines are scheduled for Day 3+ must report status=NOT_IMPLEMENTED.
    """
    registry = CapabilityRegistry(runtime_mode=RuntimeMode.DEMO_FALLBACK)
    all_caps = registry.get_all()

    # Core data gates are READY and EXECUTED
    assert all_caps["upload"]["status"] == "READY"
    assert all_caps["metadata_inspection"]["status"] == "READY"
    assert all_caps["compatibility_check"]["status"] == "READY"

    # Analysis capabilities: routing is READY, execution is NOT_IMPLEMENTED
    for cap_name in ("single_image_vqa_sar", "single_image_vqa_optical", "temporal_change", "optical_sar_fusion"):
        cap = all_caps[cap_name]
        assert cap["status"] == "NOT_IMPLEMENTED"
        assert cap["routing_readiness"] == "READY"
        assert "NOT_IMPLEMENTED" in cap["execution_readiness"]


def test_dataset_provenance_and_governance_rules():
    """
    Verify dataset provenance tracking and governance rules:
    1. InputSource and DatasetRole correctly detected from raster source.
    2. Training datasets must never automatically become evaluation datasets.
    3. VRSBench evaluation data must never be used for fine-tuning.
    4. RoutingDecision explicitly preserves input_sources and dataset_roles.
    """
    from src.contracts.raster_contracts import (
        InputSource,
        DatasetRole,
        validate_dataset_governance,
    )
    from src.gateway.raster_inspector import RasterInspector

    inspector = RasterInspector()

    # Provenance detection heuristics
    src, role = inspector._detect_input_source(Path("vrsbench_eval_sample_01.tif"))
    assert src == InputSource.VRSBENCH
    assert role == DatasetRole.BENCHMARK_EVALUATION

    src, role = inspector._detect_input_source(Path("bigearthnet_s2_patch_04.tif"))
    assert src == InputSource.BIGEARTHNET_TXT
    assert role == DatasetRole.TRAINING

    src, role = inspector._detect_input_source(Path("data/samples/optical/synthetic_optical_rgb.tif"))
    assert src == InputSource.SYNTHETIC_ENGINEERING
    assert role == DatasetRole.VALIDATION

    src, role = inspector._detect_input_source(Path("my_field_survey.tif"))
    assert src == InputSource.USER_UPLOAD
    assert role == DatasetRole.INFERENCE

    # Governance rule 1: VRSBench must never be used for fine-tuning/training
    valid, err = validate_dataset_governance(InputSource.VRSBENCH, DatasetRole.TRAINING)
    assert valid is False
    assert "VRSBench evaluation data must never be used for fine-tuning" in err

    # Governance rule 2: Training datasets (BigEarthNet) must never automatically become evaluation datasets
    valid, err = validate_dataset_governance(InputSource.BIGEARTHNET_TXT, DatasetRole.BENCHMARK_EVALUATION)
    assert valid is False
    assert "must never automatically become benchmark evaluation datasets" in err

    # Legitimate assignments pass
    valid, err = validate_dataset_governance(InputSource.VRSBENCH, DatasetRole.BENCHMARK_EVALUATION)
    assert valid is True
    assert err is None

    valid, err = validate_dataset_governance(InputSource.BIGEARTHNET_TXT, DatasetRole.TRAINING)
    assert valid is True
    assert err is None

    # End-to-end API upload & query preserves provenance
    opt_path = SAMPLES_DIR / "optical" / "synthetic_optical_rgb.tif"
    with open(opt_path, "rb") as f:
        up_resp = client.post("/api/upload", files={"file": ("synthetic_optical_rgb.tif", f, "image/tiff")})
    assert up_resp.status_code == 200
    meta = up_resp.json()["metadata"]
    assert meta["input_source"] == InputSource.SYNTHETIC_ENGINEERING.value
    assert meta["dataset_role"] == DatasetRole.VALIDATION.value

    file_id = up_resp.json()["file_id"]
    q_resp = client.post(
        "/api/query",
        json={"query": "Identify land cover features", "file_ids": [file_id]},
    )
    assert q_resp.status_code == 200
    decision = q_resp.json()["decision"]
    assert InputSource.SYNTHETIC_ENGINEERING.value in decision["input_sources"]
    assert DatasetRole.VALIDATION.value in decision["dataset_roles"]

