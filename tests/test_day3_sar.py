"""
SatQuery AI -- Day 3 Deterministic SAR Analysis Tests
Verifies:
- test_sar_backscatter_statistics
- test_sar_linear_conversion
- test_sar_vv_vh_linear_ratio
- test_sar_vv_vh_db_difference
- test_lee_filter_linear_domain
- test_lee_filter_nodata
- test_sar_water_adaptive_threshold
- test_sar_structured_response
- test_sar_engine_analyze_raster
"""

import math
from pathlib import Path
import numpy as np
import pytest

from src.analysis.sar_tools import (
    DeterministicSAREngine,
    LeeSpeckleFilter,
    PolarizationRatioEstimator,
    SARBackscatterAnalysis,
    SARStructuredResponseComposer,
    SARWaterDetector,
)
from src.analysis.numerical_math import (
    compute_sar_polarization_difference_db,
    compute_sar_polarization_ratio_linear,
    sar_db_to_linear,
    sar_linear_to_db,
)
from src.contracts.query_contracts import TaskType

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_sar_backscatter_statistics():
    """Verify statistics computation on known synthetic values with nodata."""
    analyzer = SARBackscatterAnalysis()
    # Array with known values: [-10, -12, -14, -16], mean = -13, min = -16, max = -10
    arr = np.array([[-10.0, -12.0], [-14.0, -16.0]], dtype=np.float64)
    stats = analyzer.analyze(arr, polarization="VV", input_domain="dB")

    assert stats.polarization == "VV"
    assert stats.valid_pixels == 4
    assert stats.total_pixels == 4
    assert stats.nodata_count == 0
    assert math.isclose(stats.mean, -13.0, abs_tol=1e-5)
    assert math.isclose(stats.min, -16.0, abs_tol=1e-5)
    assert math.isclose(stats.max, -10.0, abs_tol=1e-5)

    # Test with NaN and nodata value
    arr_nodata = np.array([[-10.0, np.nan], [-9999.0, -14.0]], dtype=np.float64)
    stats_nd = analyzer.analyze(arr_nodata, polarization="VH", input_domain="dB", nodata=-9999.0)
    assert stats_nd.valid_pixels == 2
    assert stats_nd.nodata_count == 2
    assert math.isclose(stats_nd.mean, -12.0, abs_tol=1e-5)


def test_sar_linear_conversion():
    """Verify conversion from decibels to linear power: 10^(dB/10) and roundtrip."""
    # -10 dB -> 10^(-1) = 0.1 linear power
    db_val = -10.0
    lin_val = sar_db_to_linear(db_val)
    assert math.isclose(lin_val, 0.1, rel_tol=1e-5)

    # 0 dB -> 1.0 linear power
    assert math.isclose(sar_db_to_linear(0.0), 1.0, rel_tol=1e-5)

    # -20 dB -> 0.01 linear power
    assert math.isclose(sar_db_to_linear(-20.0), 0.01, rel_tol=1e-5)

    # Roundtrip check
    assert math.isclose(sar_linear_to_db(lin_val), db_val, abs_tol=1e-5)


def test_sar_vv_vh_linear_ratio():
    """
    Verify scientifically correct polarization ratio:
    VV_dB = -10 dB, VH_dB = -15 dB
    Linear ratio = 10^((-10 - (-15))/10) = 10^(0.5) = sqrt(10) ≈ 3.162277
    """
    # Scalar inputs
    ratio_scalar = compute_sar_polarization_ratio_linear(-10.0, -15.0)
    expected = 10.0 ** 0.5
    assert math.isclose(float(ratio_scalar), expected, rel_tol=1e-4)

    # Array inputs
    vv_db = np.array([-10.0])
    vh_db = np.array([-15.0])
    ratio_arr = compute_sar_polarization_ratio_linear(vv_db, vh_db)
    assert math.isclose(float(ratio_arr[0]), expected, rel_tol=1e-4)


def test_sar_vv_vh_db_difference():
    """
    Verify cross-polarization difference in dB:
    VV_dB = -10 dB, VH_dB = -15 dB
    Difference = -10 - (-15) = 5.0 dB
    """
    # Scalar input
    diff_scalar = compute_sar_polarization_difference_db(-10.0, -15.0)
    assert math.isclose(float(diff_scalar), 5.0, abs_tol=1e-5)

    # Array input
    vv_db = np.array([-10.0])
    vh_db = np.array([-15.0])
    diff_arr = compute_sar_polarization_difference_db(vv_db, vh_db)
    assert math.isclose(float(diff_arr[0]), 5.0, abs_tol=1e-5)


def test_lee_filter_linear_domain():
    """
    Verify Lee speckle filter operates in linear domain:
    Input in dB -> converted to linear -> smoothed -> converted to dB.
    Smoothes variance while preserving global mean.
    """
    lee = LeeSpeckleFilter(window_size=5, enl=4.0)

    # Create synthetic noisy region around -12 dB with speckle noise
    np.random.seed(42)
    clean_db = -12.0
    clean_linear = 10.0 ** (clean_db / 10.0)
    # Multiplicative speckle: Gamma distributed intensity with shape=ENL=4
    speckle = np.random.gamma(shape=4.0, scale=0.25, size=(64, 64))
    noisy_linear = clean_linear * speckle
    noisy_db = 10.0 * np.log10(np.maximum(noisy_linear, 1e-6))

    var_before = float(np.var(noisy_db))
    res = lee.filter(noisy_db, input_is_db=True, output_as_db=True)
    var_after = float(np.var(res.filtered_array))

    # Variance should be significantly reduced by speckle filtering
    assert var_after < var_before
    assert res.input_domain == "dB"
    assert res.output_domain == "dB"
    assert len(res.assumptions) >= 2


def test_lee_filter_nodata():
    """Verify Lee filter preserves NaN/nodata locations."""
    lee = LeeSpeckleFilter(window_size=3, enl=4.0)
    arr = np.full((10, 10), -12.0)
    arr[0, 0] = -9999.0
    arr[5, 5] = np.nan

    res = lee.filter(arr, input_is_db=True, output_as_db=True, nodata=-9999.0)
    out = res.filtered_array
    assert out[0, 0] == -9999.0
    assert np.isnan(out[5, 5])
    # Valid pixels should remain around -12 dB
    assert math.isclose(float(out[2, 2]), -12.0, abs_tol=0.1)


def test_sar_water_adaptive_threshold():
    """
    Verify adaptive water candidate detector:
    Water features typically display low backscatter (<-18 dB) due to specular reflection.
    Land displays higher backscatter (>-13 dB).
    """
    detector = SARWaterDetector()

    # Synthetic scene: 20% water (-22 dB), 80% land (-10 dB)
    arr = np.full((50, 50), -10.0)
    arr[:20, :] = -22.0  # 40% water region

    res = detector.detect(arr, polarization="VV", is_db=True, apply_filter=False)

    assert res.water_pixels == 20 * 50
    assert math.isclose(res.water_fraction, 0.40, abs_tol=0.01)
    assert res.threshold_value_db < -12.0
    assert res.threshold_value_db > -22.0
    assert "adaptive_otsu" in res.threshold_method
    assert len(res.limitations) >= 3


def test_sar_structured_response():
    """Verify SARStructuredResponseComposer produces factual, non-hallucinated response."""
    composer = SARStructuredResponseComposer()
    analyzer = SARBackscatterAnalysis()
    ratio_est = PolarizationRatioEstimator()
    water_det = SARWaterDetector()

    vv_arr = np.full((30, 30), -11.5)
    vh_arr = np.full((30, 30), -17.5)

    stats_vv = analyzer.analyze(vv_arr, polarization="VV")
    stats_vh = analyzer.analyze(vh_arr, polarization="VH")
    ratio = ratio_est.estimate(vv_arr, vh_arr)
    water = water_det.detect(vv_arr, polarization="VV", apply_filter=False)

    res = composer.compose(
        query="What is the average radar backscatter?",
        stats_vv=stats_vv,
        stats_vh=stats_vh,
        ratio=ratio,
        water=water,
        lee_filter_info=None,
    )

    assert res.task == TaskType.SINGLE_IMAGE_VQA_SAR
    assert res.mechanism == "deterministic_sar_analysis"
    assert res.model is None
    assert res.status == "EXECUTED"
    assert "-11.5" in res.answer
    assert "-17.5" in res.answer
    assert "6.0 dB" in res.answer or "6.0" in res.answer


def test_sar_engine_analyze_raster():
    """Verify DeterministicSAREngine runs full pipeline on synthetic SAR GeoTIFF."""
    engine = DeterministicSAREngine()
    sar_path = SAMPLES_DIR / "sar" / "synthetic_sar_vv_vh.tif"

    res, tools = engine.analyze_raster(sar_path, query="Analyze radar roughness")

    assert res.status == "EXECUTED"
    assert res.mechanism == "deterministic_sar_analysis"
    assert res.answer is not None
    assert len(res.evidence) >= 2
    assert "RasterInspector" in tools
    assert "SARBackscatterAnalysis" in tools
    assert "LeeSpeckleFilter" in tools
    assert "PolarizationRatioEstimator" in tools
    assert "SARWaterDetector" in tools
    assert "SARStructuredResponseComposer" in tools
