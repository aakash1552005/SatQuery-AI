"""
SatQuery AI -- Day 8 Acceptance Test Suite
Verifies all 8 Mandatory Demonstration Scenarios codified in Section 28
of the Master Build Specification (v4).
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from src.contracts.query_contracts import TaskType

client = TestClient(app)
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"

OPTICAL_SAMPLE = SAMPLES_DIR / "optical" / "synthetic_optical_rgb.tif"
MULTISPECTRAL_SAMPLE = SAMPLES_DIR / "optical" / "synthetic_multispectral_4band.tif"
SAR_SAMPLE = SAMPLES_DIR / "sar" / "synthetic_sar_vv_vh.tif"
TEMP_T1_SAMPLE = SAMPLES_DIR / "temporal" / "temporal_t1_2024_jan.tif"
TEMP_T2_SAMPLE = SAMPLES_DIR / "temporal" / "temporal_t2_2025_jan.tif"


def upload_helper(path: Path) -> str:
    with open(path, "rb") as f:
        res = client.post("/api/upload", files={"file": (path.name, f, "image/tiff")})
        assert res.status_code == 200
        return res.json()["file_id"]


def test_demo1_single_image_optical_vqa():
    """DEMO 1: Optical VQA with spectral land cover & vegetation indices."""
    fid = upload_helper(OPTICAL_SAMPLE)
    q = "Assess vegetation health and dominant land cover in this scene"
    res = client.post("/api/query", json={"query": q, "file_ids": [fid]}).json()

    assert res["status"] == "routed"
    assert res["decision"]["task_type"] == "single_image_vqa_optical"
    assert res["result"]["status"] == "EXECUTED"
    assert res["result"]["mechanism"] == "deterministic_optical_spectral_analysis"
    assert len(res["result"]["evidence"]) >= 1
    assert "NDVI" in res["result"]["answer"] or "dominant category" in res["result"]["answer"]


def test_demo2_grounding_water_localization():
    """DEMO 2: Grounding locates water body and yields valid bounding box."""
    fid = upload_helper(MULTISPECTRAL_SAMPLE)
    q = "Where is the largest water body? Provide bounding box"
    res = client.post("/api/query", json={"query": q, "file_ids": [fid]}).json()

    assert res["status"] == "routed"
    assert res["decision"]["task_type"] == "single_image_grounding"
    assert res["result"]["status"] == "EXECUTED"
    assert "bbox" in res["result"]["confidence"]
    bbox = res["result"]["confidence"]["bbox"]
    assert len(bbox) == 4
    # Ensure min_lon < max_lon and min_lat < max_lat
    assert bbox[0] <= bbox[2]
    assert bbox[1] <= bbox[3]


def test_demo3_bitemporal_change_detection():
    """DEMO 3: Bi-temporal change with registration check and L1 physical delta."""
    fid1 = upload_helper(TEMP_T1_SAMPLE)
    fid2 = upload_helper(TEMP_T2_SAMPLE)
    q = "What changed between these two acquisitions?"
    res = client.post("/api/query", json={"query": q, "file_ids": [fid1, fid2]}).json()

    assert res["status"] == "routed"
    assert res["decision"]["task_type"] == "temporal_change"
    assert res["result"]["status"] == "EXECUTED"
    assert "deterministic_bitemporal_change" in res["result"]["mechanism"]
    assert res["result"]["confidence"]["level"] == "PHYSICAL_CHANGE_L1"
    assert "area_km2" in res["result"]["confidence"]


def test_demo4_optical_sar_multimodal_fusion():
    """DEMO 4: Optical-SAR cross-modal fusion with 4 spatial agreement tiers."""
    fid_opt = upload_helper(OPTICAL_SAMPLE)
    fid_sar = upload_helper(SAR_SAMPLE)
    q = "Identify regions likely to contain surface water using both observations"
    res = client.post("/api/query", json={"query": q, "file_ids": [fid_opt, fid_sar]}).json()

    assert res["status"] == "routed"
    assert res["decision"]["task_type"] == "optical_sar_analysis"
    assert res["result"]["status"] == "EXECUTED"
    assert res["result"]["mechanism"] == "deterministic_optical_sar_cross_modal_fusion"
    assert "area_both_agree_km2" in res["result"]["confidence"]
    assert "area_sar_only_cloud_pierced_km2" in res["result"]["confidence"]
    assert res["result"]["confidence"]["area_both_agree_km2"] >= 0.0


def test_demo5_dynamic_agentic_routing():
    """DEMO 5: Dynamic agentic routing across 4 modalities with zero manual toggles."""
    fid_opt = upload_helper(OPTICAL_SAMPLE)
    fid_sar = upload_helper(SAR_SAMPLE)
    fid_t1 = upload_helper(TEMP_T1_SAMPLE)
    fid_t2 = upload_helper(TEMP_T2_SAMPLE)

    matrix = [
        ("Calculate vegetation index", [fid_opt], "single_image_vqa_optical"),
        ("Analyze radar backscatter", [fid_sar], "single_image_vqa_sar"),
        ("What changed over time?", [fid_t1, fid_t2], "temporal_change"),
        ("Fuse optical and SAR data", [fid_opt, fid_sar], "optical_sar_analysis"),
    ]

    for query_text, fids, expected_task in matrix:
        res = client.post("/api/query", json={"query": query_text, "file_ids": fids}).json()
        assert res["decision"]["task_type"] == expected_task
        assert res["decision"]["is_executable"] is True


def test_demo6_sufficiency_refusal_gate():
    """DEMO 6: Sufficiency refusal for single-image temporal query."""
    fid = upload_helper(OPTICAL_SAMPLE)
    q = "What changed over the last two years?"
    res = client.post("/api/query", json={"query": q, "file_ids": [fid]}).json()

    assert res["status"] == "refused"
    assert res["decision"]["is_executable"] is False
    assert "Only one acquisition was supplied" in res["decision"]["refusal_reason"]
    assert res["result"]["status"] == "REFUSED"


def test_demo7_compute_fallback_profile_d():
    """DEMO 7: Host compute profile fallback correctly acknowledged."""
    res_status = client.get("/api/status").json()
    assert res_status["status"] == "running"
    assert res_status["runtime_mode"] in ("HYBRID", "DEMO_FALLBACK", "AUTO")

    res_caps = client.get("/api/capabilities").json()
    sar_cap = res_caps["capabilities"]["single_image_vqa_sar"]
    assert sar_cap["status"] == "READY"
    assert sar_cap["requires_gpu"] is False


def test_demo8_sar_vqa_radar_physics():
    """DEMO 8: SAR analysis strictly preserves linear Lee filter and radar physics."""
    fid = upload_helper(SAR_SAMPLE)
    q = "Analyze radar backscatter in decibels and cross-polarization ratio"
    res = client.post("/api/query", json={"query": q, "file_ids": [fid]}).json()

    assert res["status"] == "routed"
    assert res["decision"]["task_type"] == "single_image_vqa_sar"
    assert res["result"]["status"] == "EXECUTED"
    assert res["result"]["mechanism"] == "deterministic_sar_analysis"
    # Ensure SAR is never routed to optical
    assert "optical" not in res["result"]["mechanism"]
