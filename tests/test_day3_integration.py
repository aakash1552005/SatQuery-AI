"""
SatQuery AI -- Day 3 Integration & End-to-End Execution Tests
Verifies:
- test_sar_query_executes_tools
- test_optical_query_executes_tools
- test_execution_trace_contains_actual_tool_execution
- test_capability_registry_updates_after_implementation
- test_dataset_registry_manifest
- test_geochat_dynamic_preflight
"""

from pathlib import Path
import yaml
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from src.contracts.query_contracts import TaskType, ToolExecutionStatus
from src.router.capability_registry import CapabilityRegistry, RuntimeMode

client = TestClient(app)
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "data" / "manifests"


def test_sar_query_executes_tools():
    """Verify live /api/query for SAR raster executes actual tools."""
    sar_path = SAMPLES_DIR / "sar" / "synthetic_sar_vv_vh.tif"
    with open(sar_path, "rb") as f:
        up_resp = client.post("/api/upload", files={"file": ("sar_int_test.tif", f, "image/tiff")})
    assert up_resp.status_code == 200
    file_id = up_resp.json()["file_id"]

    q_resp = client.post(
        "/api/query",
        json={"query": "Perform radar backscatter and water detection", "file_ids": [file_id]},
    )
    assert q_resp.status_code == 200
    data = q_resp.json()

    # Decision checks
    assert data["decision"]["task_type"] == TaskType.SINGLE_IMAGE_VQA_SAR
    assert data["decision"]["is_executable"] is True

    # Real tool execution checks
    tool_execs = {t["tool_name"]: t["status"] for t in data["decision"]["tool_executions"]}
    assert tool_execs["RasterInspector"] == ToolExecutionStatus.EXECUTED.value

    # Real analysis result
    res = data["result"]
    assert res["status"] == "EXECUTED"
    assert res["mechanism"] == "deterministic_sar_analysis"
    assert res["model"] is None
    assert res["answer"] is not None
    assert "dB" in res["answer"]


def test_optical_query_executes_tools():
    """Verify live /api/query for optical raster executes actual tools."""
    ms_path = SAMPLES_DIR / "optical" / "synthetic_multispectral_4band.tif"
    with open(ms_path, "rb") as f:
        up_resp = client.post("/api/upload", files={"file": ("ms_int_test.tif", f, "image/tiff")})
    assert up_resp.status_code == 200
    file_id = up_resp.json()["file_id"]

    q_resp = client.post(
        "/api/query",
        json={"query": "What is the vegetation index and land cover?", "file_ids": [file_id]},
    )
    assert q_resp.status_code == 200
    data = q_resp.json()

    assert data["decision"]["task_type"] == TaskType.SINGLE_IMAGE_VQA_OPTICAL
    assert data["decision"]["is_executable"] is True

    res = data["result"]
    assert res["status"] == "EXECUTED"
    assert res["mechanism"] == "deterministic_optical_spectral_analysis"
    assert res["model"] is None
    assert res["answer"] is not None
    assert "NDVI" in res["answer"] or "classification" in res["answer"].lower()


def test_execution_trace_contains_actual_tool_execution():
    """Verify factual execution trace contains actual tool execution steps."""
    sar_path = SAMPLES_DIR / "sar" / "synthetic_sar_vv_vh.tif"
    with open(sar_path, "rb") as f:
        up_resp = client.post("/api/upload", files={"file": ("trace_sar_test.tif", f, "image/tiff")})
    assert up_resp.status_code == 200
    file_id = up_resp.json()["file_id"]

    q_resp = client.post(
        "/api/query",
        json={"query": "Measure polarization ratio", "file_ids": [file_id]},
    )
    assert q_resp.status_code == 200
    trace = q_resp.json()["trace"]

    step_names = [s["name"] for s in trace["steps"]]
    assert "Input validation" in step_names
    assert "Sensor detected" in step_names
    assert "Pathway selected" in step_names
    assert "SAR feature extraction" in step_names
    assert "Result composed" in step_names


def test_capability_registry_updates_after_implementation():
    """Verify live /api/capabilities returns READY for Day 3 capabilities."""
    cap_resp = client.get("/api/capabilities")
    assert cap_resp.status_code == 200
    caps = cap_resp.json()["capabilities"]

    # Implemented in Day 3
    assert caps["single_image_vqa_sar"]["status"] == "READY"
    assert caps["single_image_vqa_sar"]["execution_readiness"] == "READY"
    assert caps["single_image_vqa_optical"]["status"] == "READY"
    assert caps["single_image_vqa_optical"]["execution_readiness"] == "READY"

    # Future milestones remain NOT_IMPLEMENTED
    assert caps["temporal_change"]["status"] == "NOT_IMPLEMENTED"
    assert caps["optical_sar_fusion"]["status"] == "NOT_IMPLEMENTED"


def test_dataset_registry_manifest():
    """Verify machine-readable dataset registry YAML file and role separation rules."""
    reg_path = MANIFESTS_DIR / "dataset_registry.yaml"
    assert reg_path.exists()

    with open(reg_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    assert "datasets" in manifest
    datasets = manifest["datasets"]

    # synthetic engineering
    assert datasets["synthetic_engineering"]["role"] == "engineering_validation"
    assert datasets["synthetic_engineering"]["training_allowed"] is False
    assert datasets["synthetic_engineering"]["evaluation_allowed"] is False

    # BigEarthNet.txt
    assert datasets["bigearthnet_txt"]["role"] == "training_finetuning"
    assert datasets["bigearthnet_txt"]["status"] == "PLANNED"
    assert datasets["bigearthnet_txt"]["training_allowed"] is True
    assert datasets["bigearthnet_txt"]["evaluation_allowed"] is False
    assert "https://txt.bigearth.net/" in datasets["bigearthnet_txt"]["source"]["official_page"]

    # VRSBench
    assert datasets["vrsbench"]["role"] == "public_evaluation"
    assert datasets["vrsbench"]["status"] == "PLANNED"
    assert datasets["vrsbench"]["training_allowed"] is False
    assert datasets["vrsbench"]["evaluation_allowed"] is True


def test_geochat_dynamic_preflight():
    """Verify dynamic host preflight executes and truthfully detects CPU host."""
    registry = CapabilityRegistry(runtime_mode=RuntimeMode.DEMO_FALLBACK)
    preflight = registry.run_geochat_preflight()

    assert preflight["CUDA"] == "UNAVAILABLE"
    assert preflight["GPU_VRAM"] == "NOT_AVAILABLE"
    assert preflight["environment_preflight"] in ["COMPLETED", "FAILED"]
    assert preflight["environment_result"] == "UNAVAILABLE"
    assert preflight["real_model_inference"] == "NOT_EXECUTED"
    assert preflight["final_capability"] == "UNAVAILABLE"


def test_dataset_roles():
    """Verify explicit dataset roles defined in registry."""
    reg_path = MANIFESTS_DIR / "dataset_registry.yaml"
    with open(reg_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    datasets = manifest["datasets"]
    assert datasets["synthetic_engineering"]["role"] == "engineering_validation"
    assert datasets["bigearthnet_txt"]["role"] == "training_finetuning"
    assert datasets["vrsbench"]["role"] == "public_evaluation"


def test_training_evaluation_separation():
    """Verify training datasets are barred from evaluation and evaluation datasets barred from training."""
    reg_path = MANIFESTS_DIR / "dataset_registry.yaml"
    with open(reg_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    datasets = manifest["datasets"]
    # BigEarthNet.txt (training) cannot be used as evaluation benchmark
    assert datasets["bigearthnet_txt"]["training_allowed"] is True
    assert datasets["bigearthnet_txt"]["evaluation_allowed"] is False
    # VRSBench (evaluation) cannot be used for fine-tuning
    assert datasets["vrsbench"]["training_allowed"] is False
    assert datasets["vrsbench"]["evaluation_allowed"] is True


def test_no_false_zero_leakage_claim():
    """Verify that dataset registry does NOT claim zero data leakage guaranteed before audits."""
    reg_path = MANIFESTS_DIR / "dataset_registry.yaml"
    with open(reg_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        manifest = yaml.safe_load(raw_text)

    # Must NOT claim zero leakage guaranteed
    assert "ZERO_DATA_LEAKAGE" not in raw_text
    assert "Zero data leakage guaranteed" not in raw_text

    # Must represent split_validation and duplicate_audit as PENDING
    leakage = manifest["datasets"]["vrsbench"]["leakage_governance"]
    assert leakage["role_policy"] == "enforced"
    assert leakage["split_validation"] == "PENDING"
    assert leakage["duplicate_audit"] == "PENDING"


def test_cpu_host_gpu_vram_not_available():
    """Verify CPU host truthfully reports GPU_VRAM: NOT_AVAILABLE, not INSUFFICIENT."""
    registry = CapabilityRegistry()
    preflight = registry.run_geochat_preflight()
    assert preflight["GPU_VRAM"] == "NOT_AVAILABLE"
    assert preflight["GPU_VRAM"] != "INSUFFICIENT"


def test_environment_preflight_not_real_inference():
    """Verify that completing environment preflight never counts as real model inference."""
    registry = CapabilityRegistry()
    preflight = registry.run_geochat_preflight()
    assert preflight["environment_preflight"] == "COMPLETED"
    assert preflight["real_model_inference"] == "NOT_EXECUTED"

