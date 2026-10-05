"""
SatQuery AI -- Day 7 Automated Verification Suite
Tests for:
1. EvidenceStore (Section 19: claim tracking, spatial bounding boxes, RFC 7946 GeoJSON)
2. NumericalGuard (Section 21: regex audit, mathematical token locking, anti-hallucination)
3. EvidenceVerifier (Section 22: certification, provenance check, consistency audit)
4. FieldPackGenerator & API Exporter (Section 26: air-gapped zip pack, offline viewer, GeoJSON)
"""

import io
import json
import zipfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from src.contracts.query_contracts import AnalysisResult, TaskType
from src.contracts.raster_contracts import RasterMetadata, SensorModality, SpatialBounds
from src.reporting.report_generator import FieldPackGenerator
from src.verification.evidence_store import EvidenceStore, EvidenceType
from src.verification.numerical_guard import NumericalGuard
from src.verification.verifier import EvidenceVerifier, VerificationStatus

client = TestClient(app)
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_evidence_store_crud_and_geojson():
    """Verify EvidenceStore records claims and exports RFC 7946 GeoJSON."""
    store = EvidenceStore()
    item = store.add(
        claim="Floodwater detected under monsoon storm clouds",
        source="optical_sar_fusion",
        evidence_type=EvidenceType.REGION,
        confidence={"tier": "SAR_ONLY", "calibrated": False},
        inputs=["opt_01", "sar_01"],
        bbox=[91.5, 26.1, 91.8, 26.4],
        analysis_source="deterministic",
        properties={"area_km2": 11.2},
    )

    assert item.evidence_id is not None
    assert store.get(item.evidence_id) == item
    assert len(store.get_by_input("sar_01")) == 1
    assert len(store.get_by_input("nonexistent")) == 0

    # Test GeoJSON export
    geojson = store.to_geojson()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1
    feat = geojson["features"][0]
    assert feat["geometry"]["type"] == "Polygon"
    assert feat["properties"]["claim"] == "Floodwater detected under monsoon storm clouds"


def test_numerical_guard_locks_measurements():
    """Verify NumericalGuard approves certified values and catches hallucinated numbers."""
    certified = {
        "area_both_agree_km2": 7.20,
        "area_sar_only_km2": 11.20,
        "total_water_km2": 18.40,
        "gsd_x": 10.0,
        "gsd_y": 10.0,
    }

    # 1. Truthful certified text
    truthful_text = "Verified floodwater extent is 18.40 km², with 11.20 km² penetrated beneath clouds at 10.0m resolution."
    res_valid = NumericalGuard.verify_response(truthful_text, certified)
    assert res_valid.is_valid is True
    assert len(res_valid.violations) == 0

    # 2. Hallucinated text with fabricated measurements
    hallucinated_text = "Flood extent is 999.85 km² affecting 45200 homes and 327 bridges."
    res_invalid = NumericalGuard.verify_response(hallucinated_text, certified)
    assert res_invalid.is_valid is False
    assert len(res_invalid.violations) >= 2
    assert any("999.85" in v for v in res_invalid.violations)
    assert "[NUMERICAL_GUARD_ALERT:" in res_invalid.guarded_text


def test_evidence_verifier_certifies_clean_result():
    """Verify EvidenceVerifier passes legitimate deterministic results."""
    mock_meta = [
        RasterMetadata(
            filename="cartosat_pre.tif",
            format="GTiff",
            width=256,
            height=256,
            band_count=3,
            crs="EPSG:4326",
            modality=SensorModality.OPTICAL,
            is_valid=True,
        ),
        RasterMetadata(
            filename="risat_post.tif",
            format="GTiff",
            width=256,
            height=256,
            band_count=2,
            crs="EPSG:4326",
            modality=SensorModality.SAR,
            is_valid=True,
        ),
    ]

    mock_result = AnalysisResult(
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        mechanism="deterministic_optical_sar_cross_modal_fusion",
        model=None,
        answer="Total water is 18.40 km².",
        evidence=["Verified 18.40 km² water extent"],
        confidence={"evidence_confidence": "HIGH", "calibrated": False},
        limitations=["Radar runway reflection caveat"],
        status="EXECUTED",
    )

    certified_dict = {"total_water_km2": 18.40}
    ver_report = EvidenceVerifier.verify(
        result=mock_result,
        input_metadata=mock_meta,
        certified_numbers=certified_dict,
    )

    assert ver_report.is_certified is True
    assert ver_report.status == VerificationStatus.CERTIFIED_DETERMINISTIC
    assert len(ver_report.checks_failed) == 0


def test_field_pack_zip_archive_structure():
    """Verify FieldPackGenerator bundles air-gapped files into a valid ZIP archive."""
    mock_meta = [
        RasterMetadata(
            filename="opt.tif",
            format="GTiff",
            width=256,
            height=256,
            band_count=4,
            crs="EPSG:4326",
            bounds=SpatialBounds(left=91.0, bottom=26.0, right=92.0, top=27.0),
            modality=SensorModality.OPTICAL,
            is_valid=True,
        )
    ]

    mock_result = AnalysisResult(
        task=TaskType.SINGLE_IMAGE_VQA_OPTICAL,
        mechanism="deterministic_optical_spectral_analysis",
        model=None,
        answer="Vegetation index NDVI is 0.65.",
        evidence=["NDVI mean: 0.65"],
        confidence={"type": "heuristic", "calibrated": False},
        status="EXECUTED",
    )

    zip_bytes = FieldPackGenerator.create_field_pack_bytes(
        query="Assess vegetation health",
        result=mock_result,
        trace_data={"steps": [{"name": "Inspection", "status": "completed"}]},
        input_metadata=mock_meta,
    )

    assert len(zip_bytes) > 0

    # Unpack in-memory and verify file contents
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        names = z.namelist()
        assert "evidence.geojson" in names
        assert "result.json" in names
        assert "execution_trace.json" in names
        assert "mission_intelligence_brief.html" in names
        assert "offline_field_viewer.html" in names
        assert "README_FIELD_PACK.txt" in names

        # Verify GeoJSON valid JSON
        geo_str = z.read("evidence.geojson").decode("utf-8")
        geo_json = json.loads(geo_str)
        assert geo_json["type"] == "FeatureCollection"


def test_api_export_endpoints_integration():
    """Verify live FastAPI /api/export/geojson and /api/export/field-pack endpoints."""
    opt_path = SAMPLES_DIR / "optical" / "synthetic_optical_rgb.tif"
    with open(opt_path, "rb") as f:
        resp = client.post("/api/upload", files={"file": ("export_test.tif", f, "image/tiff")})
    assert resp.status_code == 200
    fid = resp.json()["file_id"]

    # 1. GeoJSON endpoint
    geo_resp = client.post(
        "/api/export/geojson",
        json={"query": "Extract water bodies", "file_ids": [fid]},
    )
    assert geo_resp.status_code == 200
    geo_data = geo_resp.json()
    assert geo_data["type"] == "FeatureCollection"

    # 2. Field pack endpoint
    pack_resp = client.post(
        "/api/export/field-pack",
        json={"query": "Export full mission pack", "file_ids": [fid]},
    )
    assert pack_resp.status_code == 200
    assert pack_resp.headers["content-type"] == "application/zip"
    assert "attachment; filename=SatQuery_FieldPack_" in pack_resp.headers.get("content-disposition", "")
    assert len(pack_resp.content) > 1000
