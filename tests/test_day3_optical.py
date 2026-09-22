"""
SatQuery AI -- Day 3 Deterministic Optical Analysis Tests
Verifies:
- test_optical_band_mapping
- test_ndvi_query_execution
- test_ndwi_query_execution
- test_mndwi_query_execution
- test_rule_based_land_cover
- test_optical_missing_band_refusal
- test_optical_engine_analyze_raster
"""

import math
from pathlib import Path
import numpy as np
import pytest

from src.analysis.numerical_math import (
    compute_mndwi,
    compute_ndvi,
    compute_ndwi,
)
from src.analysis.optical_tools import (
    DeterministicOpticalEngine,
    OpticalBandMapper,
    RuleBasedLandCoverClassifier,
    SpectralIndexEngine,
)
from src.contracts.query_contracts import TaskType

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_optical_band_mapping():
    """Verify band mapper detects spectral roles accurately."""
    mapper = OpticalBandMapper()

    # 1. Explicit Sentinel-2 style descriptions
    bnames = ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR", "B11_SWIR1"]
    bmap = mapper.map_bands(5, bnames)
    assert bmap.blue_idx == 1
    assert bmap.green_idx == 2
    assert bmap.red_idx == 3
    assert bmap.nir_idx == 4
    assert bmap.swir1_idx == 5

    # 2. 4-band RGB-NIR convention
    bmap_4 = mapper.map_bands(4, ["Red", "Green", "Blue", "NIR"])
    assert bmap_4.red_idx == 1
    assert bmap_4.green_idx == 2
    assert bmap_4.blue_idx == 3
    assert bmap_4.nir_idx == 4

    # 3. 3-band standard RGB
    bmap_3 = mapper.map_bands(3, ["band_1", "band_2", "band_3"])
    assert bmap_3.red_idx == 1
    assert bmap_3.green_idx == 2
    assert bmap_3.blue_idx == 3
    assert bmap_3.nir_idx is None


def test_ndvi_query_execution():
    """
    Verify known-value assertion for NDVI:
    NIR = 0.8, RED = 0.2
    NDVI = (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.6
    """
    nir = np.full((10, 10), 0.8)
    red = np.full((10, 10), 0.2)

    engine = SpectralIndexEngine()
    res = engine.compute_ndvi_stat(nir, red)

    assert math.isclose(res.mean, 0.6, abs_tol=1e-4)
    assert math.isclose(res.min, 0.6, abs_tol=1e-4)
    assert math.isclose(res.max, 0.6, abs_tol=1e-4)
    assert res.valid_pixels == 100
    assert res.positive_fraction == 1.0


def test_ndwi_query_execution():
    """
    Verify known-value assertion for NDWI (McFeeters):
    GREEN = 0.6, NIR = 0.2
    NDWI = (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5
    """
    green = np.full((10, 10), 0.6)
    nir = np.full((10, 10), 0.2)

    engine = SpectralIndexEngine()
    res = engine.compute_ndwi_stat(green, nir)

    assert math.isclose(res.mean, 0.5, abs_tol=1e-4)
    assert res.valid_pixels == 100
    assert res.positive_fraction == 1.0


def test_mndwi_query_execution():
    """
    Verify known-value assertion for MNDWI (Xu):
    GREEN = 0.6, SWIR = 0.2
    MNDWI = (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5
    """
    green = np.full((10, 10), 0.6)
    swir = np.full((10, 10), 0.2)

    engine = SpectralIndexEngine()
    res = engine.compute_mndwi_stat(green, swir)

    assert math.isclose(res.mean, 0.5, abs_tol=1e-4)
    assert res.valid_pixels == 100


def test_rule_based_land_cover():
    """
    Verify rule-based land cover classification on synthetic known pixels:
    Region A (50 pixels): NDVI = 0.75 -> DENSE_VEGETATION
    Region B (50 pixels): NDWI = 0.6, NDVI = 0.0 -> WATER
    """
    classifier = RuleBasedLandCoverClassifier()

    ndvi = np.zeros((10, 10), dtype=np.float64)
    ndwi = np.zeros((10, 10), dtype=np.float64)

    # Top half: vegetation
    ndvi[:5, :] = 0.75
    ndwi[:5, :] = -0.30

    # Bottom half: water
    ndvi[5:, :] = -0.10
    ndwi[5:, :] = 0.45

    res = classifier.classify(ndvi=ndvi, ndwi=ndwi)

    assert res.total_valid_pixels == 100
    assert res.pixel_counts["DENSE_VEGETATION"] == 50
    assert res.pixel_counts["WATER"] == 50
    assert res.classes["DENSE_VEGETATION"] == 50.0
    assert res.classes["WATER"] == 50.0
    assert len(res.rules_applied) >= 4


def test_optical_missing_band_refusal():
    """Verify that an optical 3-band raster without NIR computes visible heuristics without crashing."""
    engine = DeterministicOpticalEngine()
    opt_path = SAMPLES_DIR / "optical" / "synthetic_optical_rgb.tif"

    res, tools = engine.analyze_raster(opt_path, query="What is the land cover?")

    assert res.status == "EXECUTED"
    assert res.mechanism == "deterministic_optical_spectral_analysis"
    assert res.answer is not None
    assert "RasterInspector" in tools
    assert "BandMappingValidator" in tools
    assert "RuleBasedLandCoverClassifier" in tools


def test_optical_engine_analyze_raster():
    """Verify DeterministicOpticalEngine runs full pipeline on 4-band multispectral GeoTIFF."""
    engine = DeterministicOpticalEngine()
    ms_path = SAMPLES_DIR / "optical" / "synthetic_multispectral_4band.tif"

    res, tools = engine.analyze_raster(ms_path, query="Calculate NDVI and land cover")

    assert res.status == "EXECUTED"
    assert res.mechanism == "deterministic_optical_spectral_analysis"
    assert res.answer is not None
    assert len(res.evidence) >= 2
    assert "SpectralIndexEngine (NDVI)" in tools
    assert "SpectralIndexEngine (NDWI)" in tools
    assert "RuleBasedLandCoverClassifier" in tools
    assert "OpticalStructuredResponseComposer" in tools
