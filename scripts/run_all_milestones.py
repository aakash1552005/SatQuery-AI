"""
=============================================================================
SATQUERY AI -- UNIFIED MILESTONE RUNNER (DAY 1 + DAY 2 + DAY 3)
=============================================================================
Runs all verification checks, unit tests, integration pipelines, API endpoints,
and governance policies across Day 1, Day 2, and Day 3 in a single automated flow.
Exits with code 0 on complete success, or non-zero with detailed failure diagnostics.
=============================================================================
"""

import sys
import subprocess
from pathlib import Path
import yaml
import numpy as np
from fastapi.testclient import TestClient

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from app.backend.main import app
from src.gateway.raster_inspector import RasterInspector
from src.gateway.compatibility_checker import CompatibilityChecker
from src.router.query_parser import QueryParser
from src.router.agentic_router import AgenticRouter
from src.router.capability_registry import CapabilityRegistry, run_geochat_preflight
from src.analysis.sar_tools import DeterministicSAREngine, sar_db_to_linear, sar_linear_to_db
from src.analysis.optical_tools import DeterministicOpticalEngine
from src.analysis.numerical_math import compute_ndvi, compute_ndwi

client = TestClient(app)

def banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)

def step_pass(msg: str):
    print(f"  [PASS] {msg}")

def run_all():
    print("SATQUERY AI: INITIATING COMPLETE DAY 1, 2, 3 INTEGRATION AUDIT")
    print(f"Working Directory: {root}")
    print(f"Python Executable: {sys.executable}")

    # ---------------------------------------------------------
    # 1. SYSTEM & HARDWARE PROFILE CHECK
    # ---------------------------------------------------------
    banner("Phase 1: Host System & Compute Profile")
    res_health = client.get("/api/health")
    assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
    health = res_health.json()
    assert health["version"] == "0.3.0-day3", f"Version mismatch: {health['version']}"
    step_pass(f"/api/health: status={health['status']}, version={health['version']}")

    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    status_info = res_status.json()
    step_pass(f"/api/status: runtime_mode={status_info['runtime_mode']}, status={status_info['status']}")

    geochat_audit = run_geochat_preflight()
    assert geochat_audit.get("GPU_VRAM") == "NOT_AVAILABLE" or geochat_audit.get("gpu_vram") == "NOT_AVAILABLE"
    assert geochat_audit.get("real_model_inference") == "NOT_EXECUTED"
    step_pass("GeoChat Truthful Preflight: Correctly reports GPU_VRAM: NOT_AVAILABLE on Profile D CPU host")

    # ---------------------------------------------------------
    # 2. DAY 1: GIS GATEWAY & RASTER COMPATIBILITY
    # ---------------------------------------------------------
    banner("Phase 2: Day 1 GIS Gateway & Pair Compatibility")
    samples_dir = root / "data" / "samples"
    optical_sample = samples_dir / "optical" / "synthetic_optical_rgb.tif"
    sar_sample = samples_dir / "sar" / "synthetic_sar_vv_vh.tif"
    temp_t1 = samples_dir / "temporal" / "temporal_t1_2024_jan.tif"
    temp_t2 = samples_dir / "temporal" / "temporal_t2_2025_jan.tif"

    for p in [optical_sample, sar_sample, temp_t1, temp_t2]:
        assert p.exists(), f"Sample file missing: {p}"

    inspector = RasterInspector()
    meta_opt = inspector.inspect(optical_sample)
    assert meta_opt.is_valid is True
    assert meta_opt.modality.value == "optical"
    step_pass(f"RasterInspector (Optical): {meta_opt.width}x{meta_opt.height}, {meta_opt.band_count} bands, CRS={meta_opt.crs}")

    meta_sar = inspector.inspect(sar_sample)
    assert meta_sar.is_valid is True
    assert meta_sar.modality.value == "sar"
    assert meta_sar.polarization.value == "vv_vh"
    step_pass(f"RasterInspector (SAR): {meta_sar.width}x{meta_sar.height}, polarization={meta_sar.polarization.value}")

    checker = CompatibilityChecker()
    meta_t1 = inspector.inspect(temp_t1)
    meta_t2 = inspector.inspect(temp_t2)
    compat = checker.check_pair(meta_t1, meta_t2)
    assert compat.crs_match is True
    assert compat.bounds_overlap > 0.95
    step_pass(f"CompatibilityChecker: CRS match={compat.crs_match}, Overlap={round(compat.bounds_overlap*100, 1)}%")

    # ---------------------------------------------------------
    # 3. DAY 2: QUERY PARSER, AGENTIC ROUTER, SEPARATION & REFUSALS
    # ---------------------------------------------------------
    banner("Phase 3: Day 2 Agentic Router & Refusal Gates")
    parser = QueryParser()
    intent_sar = parser.parse("Analyze radar backscatter in this Sentinel-1 image")
    assert intent_sar.task_type.value == "single_image_vqa_sar"
    step_pass(f"QueryParser (SAR intent): detected {intent_sar.task_type.value}")

    intent_temp = parser.parse("What changed between these two dates?")
    assert intent_temp.task_type.value == "temporal_change"
    step_pass(f"QueryParser (Temporal intent): detected {intent_temp.task_type.value}")

    router = AgenticRouter()
    # SAR separation rule test
    dec_sar = router.route("Analyze backscatter", [meta_sar])
    assert dec_sar.pathway.value == "sar_deterministic_tools"
    step_pass(f"SAR Separation: SAR input strictly routed to '{dec_sar.pathway.value}' (never optical VLM)")

    # Refusal test (DEMO 6): 1 image for temporal change
    dec_refusal = router.route("What changed?", [meta_opt])
    assert dec_refusal.is_executable is False
    assert dec_refusal.pathway.value == "refusal"
    assert "Only one acquisition was supplied" in dec_refusal.refusal_reason
    step_pass(f"Refusal Gate (DEMO 6): Single-image temporal query honestly refused: '{dec_refusal.refusal_reason}'")

    # Refusal test: Optical-SAR fusion with single modality
    dec_fuse_refusal = router.route("Fuse optical and SAR data", [meta_opt])
    assert dec_fuse_refusal.is_executable is False
    step_pass("Refusal Gate: Fusion query without dual modalities honestly refused")

    # ---------------------------------------------------------
    # 4. DAY 3: DETERMINISTIC ENGINES, DATASET GOVERNANCE & EXECUTION
    # ---------------------------------------------------------
    banner("Phase 4: Day 3 Deterministic Engines & Real Query Execution")
    sar_engine = DeterministicSAREngine()
    sar_res, sar_tools = sar_engine.analyze_raster(sar_sample, query="Check water and backscatter")
    assert sar_res.status == "EXECUTED"
    assert sar_res.mechanism == "deterministic_sar_analysis"
    assert "SARBackscatterAnalysis" in sar_tools
    assert "LeeSpeckleFilter" in sar_tools
    assert "PolarizationRatioEstimator" in sar_tools
    assert "SARWaterDetector" in sar_tools
    assert len(sar_res.answer) > 20
    step_pass(f"Deterministic SAR Engine: executed {len(sar_tools)} tools, generated factual answer ({len(sar_res.answer)} chars)")

    # Physics test: dB to linear roundtrip
    test_db = np.array([-20.0, -10.0, -5.0])
    test_lin = sar_db_to_linear(test_db)
    test_db_rt = sar_linear_to_db(test_lin)
    assert np.allclose(test_db, test_db_rt, atol=1e-5)
    step_pass("SAR Physics Contract: Decibel <-> Linear Power roundtrip mathematically exact")

    # Optical engine test
    opt_engine = DeterministicOpticalEngine()
    opt_res, opt_tools = opt_engine.analyze_raster(optical_sample, query="Evaluate land cover")
    assert opt_res.status == "EXECUTED"
    assert opt_res.mechanism == "deterministic_optical_spectral_analysis"
    assert len(opt_res.answer) > 20
    step_pass(f"Deterministic Optical Engine: executed {len(opt_tools)} tools, generated factual answer ({len(opt_res.answer)} chars)")

    # Numerical guard test
    zero_arr = np.zeros((10, 10))
    safe_ndvi = compute_ndvi(zero_arr, zero_arr)
    assert np.all(safe_ndvi == 0.0)
    step_pass("Optical Physics Contract: compute_ndvi protected against zero-denominator division")

    # Dataset Registry Governance test
    reg_file = root / "data" / "manifests" / "dataset_registry.yaml"
    with open(reg_file, "r", encoding="utf-8") as f:
        registry_data = yaml.safe_load(f)
    assert registry_data["datasets"]["bigearthnet_txt"]["training_allowed"] is True
    assert registry_data["datasets"]["bigearthnet_txt"]["evaluation_allowed"] is False
    assert registry_data["datasets"]["vrsbench"]["training_allowed"] is False
    assert registry_data["datasets"]["vrsbench"]["evaluation_allowed"] is True
    step_pass("Dataset Governance: Verified strict training/evaluation separation between BigEarthNet.txt and VRSBench")

    # ---------------------------------------------------------
    # 5. END-TO-END FASTAPI API TEST
    # ---------------------------------------------------------
    banner("Phase 5: Live API Upload & Query Flow")
    with open(sar_sample, "rb") as f:
        up_sar = client.post("/api/upload", files={"file": ("sar.tif", f, "image/tiff")}).json()
    with open(optical_sample, "rb") as f:
        up_opt = client.post("/api/upload", files={"file": ("opt.tif", f, "image/tiff")}).json()

    # Query SAR
    q_sar = client.post("/api/query", json={
        "query": "Analyze radar backscatter in this scene",
        "file_ids": [up_sar["file_id"]]
    }).json()
    assert q_sar["decision"]["task_type"] == "single_image_vqa_sar"
    assert q_sar["result"]["status"] == "EXECUTED"
    assert q_sar["result"]["mechanism"] == "deterministic_sar_analysis"
    step_pass(f"/api/query (SAR): status=routed, result.status={q_sar['result']['status']}, mechanism={q_sar['result']['mechanism']}")

    # Query Optical
    q_opt = client.post("/api/query", json={
        "query": "Assess vegetation health in this optical scene",
        "file_ids": [up_opt["file_id"]]
    }).json()
    assert q_opt["decision"]["task_type"] == "single_image_vqa_optical"
    assert q_opt["result"]["status"] == "EXECUTED"
    assert q_opt["result"]["mechanism"] == "deterministic_optical_spectral_analysis"
    step_pass(f"/api/query (Optical): status=routed, result.status={q_opt['result']['status']}, mechanism={q_opt['result']['mechanism']}")

    # Check Capabilities Endpoint
    caps = client.get("/api/capabilities").json()["capabilities"]
    assert caps["single_image_vqa_sar"]["status"] == "READY"
    assert caps["single_image_vqa_optical"]["status"] == "READY"
    assert caps["temporal_change"]["status"] == "NOT_IMPLEMENTED"
    assert caps["optical_sar_fusion"]["status"] == "NOT_IMPLEMENTED"
    step_pass("/api/capabilities: Accurate multi-dimensional readiness (Day 1-3 implemented, Day 4-6 pending)")

    # ---------------------------------------------------------
    # 6. PYTEST SUITE EXECUTION (ALL TESTS)
    # ---------------------------------------------------------
    banner("Phase 6: Complete Automated Test Suite (Pytest)")
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    assert proc.returncode == 0, f"Pytest failed with exit code {proc.returncode}"
    step_pass("All pytest test cases PASSED cleanly!")

    banner("SUMMARY: ALL DAY 1, 2, 3 SYSTEMS OPERATIONAL & 100% PASSING")
    print("No errors, no regressions, no unhandled exceptions, and no broken contracts found.\n")

if __name__ == "__main__":
    run_all()
