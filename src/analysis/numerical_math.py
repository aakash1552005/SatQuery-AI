"""
SatQuery AI -- Deterministic Numerical & Geospatial Mathematics
Section 8.5 & Section 18: Core scientific equations for optical indices & SAR backscatter.
Enforces mathematical rigor:
1. Optical Normalized Difference Indices with zero-denominator guards.
2. SAR Linear Polarization Ratio: sigma0_linear = 10^(dB/10); Ratio = sigma0_VV / sigma0_VH.
   (Never divide decibels directly).
3. Documentation of Lee Speckle Filter domain (dB -> linear power -> filter -> dB).
4. Documentation of Adaptive Water Thresholding (scene distribution histogram, never a hardcoded universal law).
"""

from __future__ import annotations
import numpy as np


def compute_ndvi(nir: np.ndarray | float, red: np.ndarray | float) -> np.ndarray | float:
    """
    Normalized Difference Vegetation Index (NDVI).
    NDVI = (NIR - RED) / (NIR + RED)
    Protected against division by zero.
    """
    nir_arr = np.asarray(nir, dtype=np.float64)
    red_arr = np.asarray(red, dtype=np.float64)
    denominator = nir_arr + red_arr
    numerator = nir_arr - red_arr

    # Guard zero or near-zero denominator
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(np.abs(denominator) > 1e-7, numerator / denominator, 0.0)

    return float(ndvi) if np.ndim(ndvi) == 0 else ndvi


def compute_ndwi(green: np.ndarray | float, nir: np.ndarray | float) -> np.ndarray | float:
    """
    Normalized Difference Water Index (McFeeters 1996).
    NDWI = (GREEN - NIR) / (GREEN + NIR)
    Protected against division by zero.
    """
    green_arr = np.asarray(green, dtype=np.float64)
    nir_arr = np.asarray(nir, dtype=np.float64)
    denominator = green_arr + nir_arr
    numerator = green_arr - nir_arr

    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = np.where(np.abs(denominator) > 1e-7, numerator / denominator, 0.0)

    return float(ndwi) if np.ndim(ndwi) == 0 else ndwi


def compute_mndwi(green: np.ndarray | float, swir: np.ndarray | float) -> np.ndarray | float:
    """
    Modified Normalized Difference Water Index (Xu 2006).
    MNDWI = (GREEN - SWIR) / (GREEN + SWIR)
    Protected against division by zero.
    """
    green_arr = np.asarray(green, dtype=np.float64)
    swir_arr = np.asarray(swir, dtype=np.float64)
    denominator = green_arr + swir_arr
    numerator = green_arr - swir_arr

    with np.errstate(divide='ignore', invalid='ignore'):
        mndwi = np.where(np.abs(denominator) > 1e-7, numerator / denominator, 0.0)

    return float(mndwi) if np.ndim(mndwi) == 0 else mndwi


def sar_db_to_linear(val_db: np.ndarray | float) -> np.ndarray | float:
    """
    Convert SAR backscatter from decibels (dB) to linear power/intensity.
    sigma0_linear = 10^(sigma0_dB / 10)
    """
    db_arr = np.asarray(val_db, dtype=np.float64)
    linear = np.power(10.0, db_arr / 10.0)
    return float(linear) if np.ndim(linear) == 0 else linear


def sar_linear_to_db(val_linear: np.ndarray | float) -> np.ndarray | float:
    """
    Convert SAR linear power/intensity back to decibels (dB).
    sigma0_dB = 10 * log10(sigma0_linear)
    Protected against non-positive inputs.
    """
    lin_arr = np.asarray(val_linear, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        safe_lin = np.maximum(lin_arr, 1e-12)
        db = 10.0 * np.log10(safe_lin)
    return float(db) if np.ndim(db) == 0 else db


def compute_sar_polarization_ratio_linear(
    vv_db: np.ndarray | float,
    vh_db: np.ndarray | float
) -> np.ndarray | float:
    """
    Scientifically correct SAR VV/VH polarization ratio in the linear domain.
    Decibels cannot be divided directly (VV_dB / VH_dB is mathematically invalid).

    Ratio_linear = sigma0_VV_linear / sigma0_VH_linear
                 = 10^((VV_dB - VH_dB) / 10)
    """
    vv_arr = np.asarray(vv_db, dtype=np.float64)
    vh_arr = np.asarray(vh_db, dtype=np.float64)
    diff_db = vv_arr - vh_arr
    ratio = np.power(10.0, diff_db / 10.0)
    return float(ratio) if np.ndim(ratio) == 0 else ratio


def compute_sar_polarization_difference_db(
    vv_db: np.ndarray | float,
    vh_db: np.ndarray | float
) -> np.ndarray | float:
    """
    SAR cross-polarization difference in the logarithmic (dB) domain:
    Delta_dB = VV_dB - VH_dB
    (Represents the logarithmic ratio of backscatter).
    """
    vv_arr = np.asarray(vv_db, dtype=np.float64)
    vh_arr = np.asarray(vh_db, dtype=np.float64)
    diff = vv_arr - vh_arr
    return float(diff) if np.ndim(diff) == 0 else diff
