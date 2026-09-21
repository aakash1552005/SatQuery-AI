"""
SatQuery AI -- Scientific Numerical Correctness Tests
Section 8.5 & Section 18: Verification of mathematical formulas and domain rules.
Note: These tests verify numerical correctness of deterministic formulas,
NOT machine learning model accuracy or benchmark performance.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.numerical_math import (
    compute_ndvi,
    compute_ndwi,
    compute_mndwi,
    sar_db_to_linear,
    sar_linear_to_db,
    compute_sar_polarization_ratio_linear,
    compute_sar_polarization_difference_db,
)


def test_ndvi_known_values():
    """Verify NDVI known value calculation: (0.8 - 0.2) / (0.8 + 0.2) = 0.6."""
    nir = 0.8
    red = 0.2
    expected = 0.6
    result = compute_ndvi(nir, red)
    assert pytest.approx(result, abs=1e-5) == expected


def test_ndvi_zero_denominator_guard():
    """Verify that zero reflection does not cause division by zero error."""
    result = compute_ndvi(0.0, 0.0)
    assert result == 0.0


def test_ndwi_and_mndwi_known_values():
    """Verify NDWI and MNDWI known value calculations."""
    green = 0.6
    nir = 0.2
    expected_ndwi = (0.6 - 0.2) / (0.6 + 0.2)  # 0.4 / 0.8 = 0.5
    assert pytest.approx(compute_ndwi(green, nir), abs=1e-5) == expected_ndwi

    swir = 0.2
    assert pytest.approx(compute_mndwi(green, swir), abs=1e-5) == 0.5


def test_sar_linear_ratio_from_db():
    """
    Verify scientifically correct SAR polarization ratio.
    VV = -10 dB, VH = -20 dB -> Delta = +10 dB -> Ratio_linear = 10^(10/10) = 10.0.
    Directly prevents the mathematical error of dividing dB by dB (-10 / -20 = 0.5).
    """
    vv_db = -10.0
    vh_db = -20.0

    # Mathematically correct linear ratio:
    ratio_linear = compute_sar_polarization_ratio_linear(vv_db, vh_db)
    assert pytest.approx(ratio_linear, abs=1e-5) == 10.0

    # Decibel difference:
    diff_db = compute_sar_polarization_difference_db(vv_db, vh_db)
    assert pytest.approx(diff_db, abs=1e-5) == 10.0


def test_sar_db_linear_roundtrip():
    """Verify roundtrip conversion between dB and linear power domains."""
    original_db = -14.5
    linear = sar_db_to_linear(original_db)
    recovered_db = sar_linear_to_db(linear)
    assert pytest.approx(recovered_db, abs=1e-5) == original_db
