"""
SatQuery AI -- Day 1 Automated Acceptance Tests
Tests for:
1. RasterInspector (GeoTIFF parser, metadata extractor, validation gate)
2. CompatibilityChecker (CRS, resolution, footprint overlap)
3. FastAPI Backend endpoints (/api/health, /api/status, /api/upload, /api/compatibility)
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts.raster_contracts import SensorModality
from src.gateway.raster_inspector import RasterInspector
from src.gateway.compatibility_checker import CompatibilityChecker
from app.backend.main import app

client = TestClient(app)
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


def test_health_and_status():
    """Verify health and system status endpoints."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    status_resp = client.get("/api/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == "running"
    assert status_data["capabilities"]["upload"] == "READY"
    assert status_data["capabilities"]["metadata_inspection"] == "READY"
    assert status_data["capabilities"]["compatibility_check"] == "READY"


def test_raster_inspector_optical():
    """Verify raster inspection of synthetic optical RGB GeoTIFF."""
    inspector = RasterInspector()
    optical_path = SAMPLES_DIR / "optical" / "synthetic_optical_rgb.tif"
    assert optical_path.exists(), f"Sample missing: {optical_path}"

    meta = inspector.inspect(str(optical_path))
    assert meta.is_valid is True
    assert meta.width == 256
    assert meta.height == 256
    assert meta.band_count == 3
    assert meta.crs == "EPSG:4326"
    assert meta.modality == SensorModality.OPTICAL


def test_raster_inspector_multispectral():
    """Verify raster inspection of 4-band multispectral GeoTIFF."""
    inspector = RasterInspector()
    ms_path = SAMPLES_DIR / "optical" / "synthetic_multispectral_4band.tif"
    assert ms_path.exists(), f"Sample missing: {ms_path}"

    meta = inspector.inspect(str(ms_path))
    assert meta.is_valid is True
    assert meta.width == 256
    assert meta.height == 256
    assert meta.band_count == 4
    assert meta.modality == SensorModality.MULTISPECTRAL


def test_raster_inspector_sar():
    """Verify raster inspection of 2-band SAR VV/VH GeoTIFF."""
    inspector = RasterInspector()
    sar_path = SAMPLES_DIR / "sar" / "synthetic_sar_vv_vh.tif"
    assert sar_path.exists(), f"Sample missing: {sar_path}"

    meta = inspector.inspect(str(sar_path))
    assert meta.is_valid is True
    assert meta.width == 256
    assert meta.height == 256
    assert meta.band_count == 2
    assert meta.modality == SensorModality.SAR


def test_compatibility_temporal_pair():
    """Verify compatibility checker on bi-temporal optical pair."""
    inspector = RasterInspector()
    checker = CompatibilityChecker()

    t1_path = SAMPLES_DIR / "temporal" / "temporal_t1_2024_jan.tif"
    t2_path = SAMPLES_DIR / "temporal" / "temporal_t2_2025_jan.tif"

    meta_t1 = inspector.inspect(str(t1_path))
    meta_t2 = inspector.inspect(str(t2_path))

    compat = checker.check_pair(meta_t1, meta_t2)
    assert compat.is_compatible is True
    assert compat.crs_match is True
    assert compat.bounds_overlap is not None and compat.bounds_overlap > 0.9


def test_upload_and_compatibility_api():
    """Verify upload endpoint and compatibility check endpoint."""
    opt_path = SAMPLES_DIR / "optical" / "synthetic_optical_rgb.tif"
    sar_path = SAMPLES_DIR / "sar" / "synthetic_sar_vv_vh.tif"

    with open(opt_path, "rb") as f_opt:
        resp_a = client.post("/api/upload", files={"file": ("test_opt.tif", f_opt, "image/tiff")})
    assert resp_a.status_code == 200
    data_a = resp_a.json()
    assert data_a["status"] == "accepted"
    file_id_a = data_a["file_id"]

    with open(sar_path, "rb") as f_sar:
        resp_b = client.post("/api/upload", files={"file": ("test_sar.tif", f_sar, "image/tiff")})
    assert resp_b.status_code == 200
    data_b = resp_b.json()
    assert data_b["status"] == "accepted"
    file_id_b = data_b["file_id"]

    # Test compatibility endpoint
    compat_resp = client.post(
        "/api/compatibility",
        json={"file_id_a": file_id_a, "file_id_b": file_id_b},
    )
    assert compat_resp.status_code == 200
    compat_data = compat_resp.json()
    assert compat_data["status"] in ("compatible", "compatible_with_warnings")
    assert "bounds_overlap" in compat_data
