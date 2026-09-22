"""
SatQuery AI -- Deterministic SAR Analysis Engine
Section 9 (Sensor-Aware Routing) & Section 18 (SAR Analysis Tools).

Implements the factual, deterministic radar feature extraction pipeline:
1. SARBackscatterAnalysis (VV/VH statistical characterization)
2. LeeSpeckleFilter (physically correct filtering in linear power domain)
3. PolarizationRatioEstimator (linear ratio and dB difference)
4. SARWaterDetector (adaptive scene thresholding for water candidate masking)
5. SARStructuredResponseComposer (factual response without hallucination)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from src.analysis.numerical_math import (
    compute_sar_polarization_difference_db,
    compute_sar_polarization_ratio_linear,
    sar_db_to_linear,
    sar_linear_to_db,
)
from src.contracts.query_contracts import AnalysisResult, TaskType

logger = logging.getLogger("satquery.analysis.sar")


@dataclass
class BackscatterStats:
    """Statistical summary of backscatter for a single polarization band."""
    polarization: str
    input_domain: str  # "dB" or "linear_power"
    valid_pixels: int
    total_pixels: int
    nodata_count: int
    mean: float
    std: float
    min: float
    max: float
    domain_notes: str = ""


@dataclass
class SpeckleFilterResult:
    """Result of speckle filtering."""
    filtered_array: np.ndarray
    window_size: int
    equivalent_number_of_looks: float
    input_domain: str
    output_domain: str
    assumptions: list[str] = field(default_factory=list)


@dataclass
class PolarizationRatioResult:
    """Result of dual-pol VV/VH analysis."""
    vv_vh_ratio_linear: float
    vv_minus_vh_db: float
    valid_pixels: int
    calculation_method: str
    notes: str


@dataclass
class WaterDetectionResult:
    """Result of adaptive SAR water candidate detection."""
    water_mask: np.ndarray
    threshold_value_db: float
    threshold_method: str
    water_pixels: int
    total_valid_pixels: int
    water_fraction: float
    polarization_used: str
    filtering_applied: bool
    raw_threshold_db: Optional[float] = None
    accepted_threshold_db: Optional[float] = None
    threshold_adjusted: bool = False
    threshold_policy: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)


class SARBackscatterAnalysis:
    """
    Computes statistical metrics over valid pixels for a SAR polarization channel.
    Handles nodata, NaN, infinite values, and domain declarations.
    """

    def analyze(
        self,
        array: np.ndarray,
        polarization: str = "VV",
        input_domain: str = "dB",
        nodata: Optional[float] = None,
    ) -> BackscatterStats:
        arr = np.asarray(array, dtype=np.float64)
        total_pixels = int(arr.size)

        # Build validity mask
        valid_mask = np.isfinite(arr)
        if nodata is not None and not np.isnan(nodata):
            valid_mask &= (arr != nodata)

        valid_vals = arr[valid_mask]
        valid_count = int(valid_vals.size)
        nodata_count = total_pixels - valid_count

        if valid_count == 0:
            return BackscatterStats(
                polarization=polarization,
                input_domain=input_domain,
                valid_pixels=0,
                total_pixels=total_pixels,
                nodata_count=nodata_count,
                mean=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
            )

        domain_notes = (
            f"Backscatter statistics calculated in logarithmic decibel (dB) domain for {polarization}."
            if input_domain == "dB"
            else f"Backscatter statistics calculated in linear power domain for {polarization}."
        )
        return BackscatterStats(
            polarization=polarization,
            input_domain=input_domain,
            valid_pixels=valid_count,
            total_pixels=total_pixels,
            nodata_count=nodata_count,
            mean=float(np.mean(valid_vals)),
            std=float(np.std(valid_vals)),
            min=float(np.min(valid_vals)),
            max=float(np.max(valid_vals)),
            domain_notes=domain_notes,
        )


class LeeSpeckleFilter:
    """
    Physically correct Lee speckle filter for SAR imagery.
    CRITICAL PHYSICS RULE:
    Speckle is multiplicative in the linear intensity/power domain.
    Therefore, dB inputs MUST be converted to linear power before filtering,
    and converted back to dB for output.
    Preserves nodata masks and uses explicit boundary handling (reflection padding).
    """

    def __init__(self, window_size: int = 7, enl: float = 4.0):
        if window_size % 2 == 0 or window_size < 3:
            raise ValueError(f"window_size must be an odd integer >= 3, got {window_size}")
        self.window_size = window_size
        self.enl = float(enl)

    def filter(
        self,
        array: np.ndarray,
        input_is_db: bool = True,
        output_as_db: bool = True,
        nodata: Optional[float] = None,
    ) -> SpeckleFilterResult:
        arr = np.asarray(array, dtype=np.float64)
        valid_mask = np.isfinite(arr)
        if nodata is not None and not np.isnan(nodata):
            valid_mask &= (arr != nodata)

        # 1. Convert to linear power domain if input is dB
        if input_is_db:
            linear = np.where(valid_mask, sar_db_to_linear(arr), 0.0)
            in_domain = "dB"
        else:
            linear = np.where(valid_mask, np.maximum(arr, 0.0), 0.0)
            in_domain = "linear_power"

        # 2. Local moving statistics in linear power with explicit boundary reflection and NaN-normalization
        mask_f = valid_mask.astype(np.float64)
        sum_linear = uniform_filter(linear, size=self.window_size, mode="reflect")
        sum_mask = uniform_filter(mask_f, size=self.window_size, mode="reflect")
        local_mean = np.where(sum_mask > 1e-6, sum_linear / np.maximum(sum_mask, 1e-6), 0.0)

        sum_sqr = uniform_filter(linear ** 2, size=self.window_size, mode="reflect")
        local_sqr = np.where(sum_mask > 1e-6, sum_sqr / np.maximum(sum_mask, 1e-6), 0.0)
        local_var = np.maximum(0.0, local_sqr - local_mean ** 2)

        # 3. Noise variance estimate from Equivalent Number of Looks (ENL)
        noise_var_coeff = 1.0 / max(self.enl, 1.0)

        # 4. Lee adaptive weighting factor W
        noise_var_spatial = (local_mean ** 2) * noise_var_coeff
        denom = local_var + 1e-10
        weight = np.clip((local_var - noise_var_spatial) / denom, 0.0, 1.0)

        # 5. Filtered linear intensity
        filtered_linear = local_mean + weight * (linear - local_mean)
        filtered_linear = np.maximum(filtered_linear, 1e-12)

        # 6. Convert to output domain
        if output_as_db:
            out_arr = sar_linear_to_db(filtered_linear)
            out_domain = "dB"
        else:
            out_arr = filtered_linear
            out_domain = "linear_power"

        # Preserve original nodata and NaN values
        out_arr = np.where(valid_mask, out_arr, arr)

        return SpeckleFilterResult(
            filtered_array=out_arr,
            window_size=self.window_size,
            equivalent_number_of_looks=self.enl,
            input_domain=in_domain,
            output_domain=out_domain,
            assumptions=[
                f"Multiplicative speckle noise model in linear power domain (window={self.window_size}x{self.window_size})",
                f"Assumed Equivalent Number of Looks (ENL) = {self.enl} for Sentinel-1 GRD product",
                "Converted dB -> linear power -> Lee filter -> dB to maintain physical validity",
                "Preserves nodata masks and uses explicit boundary handling (scipy reflect mode)",
            ],
        )


class PolarizationRatioEstimator:
    """
    Computes polarization ratios between co-polarized and cross-polarized channels.
    Validates shapes and computes both linear ratio and dB difference.
    """

    def estimate(
        self,
        vv_array: np.ndarray,
        vh_array: np.ndarray,
        vv_is_db: bool = True,
        vh_is_db: bool = True,
        nodata: Optional[float] = None,
    ) -> PolarizationRatioResult:
        vv = np.asarray(vv_array, dtype=np.float64)
        vh = np.asarray(vh_array, dtype=np.float64)

        if vv.shape != vh.shape:
            raise ValueError(f"SAR band shape mismatch: VV {vv.shape} vs VH {vh.shape}")

        mask = np.isfinite(vv) & np.isfinite(vh)
        if nodata is not None and not np.isnan(nodata):
            mask &= (vv != nodata) & (vh != nodata)

        if not np.any(mask):
            return PolarizationRatioResult(
                vv_vh_ratio_linear=0.0,
                vv_minus_vh_db=0.0,
                valid_pixels=0,
                calculation_method="no_valid_pixels",
                notes="No valid overlapping pixels between VV and VH bands",
            )

        vv_valid = vv[mask]
        vh_valid = vh[mask]

        if not vv_is_db:
            vv_db = sar_linear_to_db(vv_valid)
        else:
            vv_db = vv_valid

        if not vh_is_db:
            vh_db = sar_linear_to_db(vh_valid)
        else:
            vh_db = vh_valid

        diff_db = compute_sar_polarization_difference_db(vv_db, vh_db)
        linear_ratio = compute_sar_polarization_ratio_linear(vv_db, vh_db)

        mean_diff = float(np.mean(diff_db))
        mean_ratio = float(np.mean(linear_ratio))

        return PolarizationRatioResult(
            vv_vh_ratio_linear=round(mean_ratio, 4),
            vv_minus_vh_db=round(mean_diff, 2),
            valid_pixels=int(vv_valid.size),
            calculation_method="sigma0_linear_ratio_and_db_difference",
            notes=(
                f"Computed linear ratio 10^((VV_dB - VH_dB)/10) = {round(mean_ratio, 4)} "
                f"and dB difference (VV - VH) = {round(mean_diff, 2)} dB"
            ),
        )


class SARWaterDetector:
    """
    Scene-adaptive water candidate detector based on SAR backscatter.
    Never relies on a fragile hardcoded constant or universal threshold claim.
    Estimates threshold adaptively from scene histogram/distribution (Otsu method),
    with explicit distinction between raw and accepted thresholds, and transparent
    engineering sanity bounds.
    """

    def __init__(
        self,
        sanity_min_db: float = -25.0,
        sanity_max_db: float = -12.0,
        enforce_sanity_bounds: bool = True,
    ):
        self.sanity_min_db = sanity_min_db
        self.sanity_max_db = sanity_max_db
        self.enforce_sanity_bounds = enforce_sanity_bounds

    def detect(
        self,
        sar_array: np.ndarray,
        polarization: str = "VV",
        is_db: bool = True,
        nodata: Optional[float] = None,
        apply_filter: bool = True,
        manual_threshold_db: Optional[float] = None,
    ) -> WaterDetectionResult:
        arr = np.asarray(sar_array, dtype=np.float64)

        if apply_filter:
            filter_engine = LeeSpeckleFilter(window_size=5, enl=4.0)
            res = filter_engine.filter(arr, input_is_db=is_db, output_as_db=True, nodata=nodata)
            proc_arr = res.filtered_array
            filtering_applied = True
        else:
            proc_arr = arr if is_db else sar_linear_to_db(arr)
            filtering_applied = False

        valid_mask = np.isfinite(proc_arr)
        if nodata is not None and not np.isnan(nodata):
            valid_mask &= (proc_arr != nodata)

        valid_vals = proc_arr[valid_mask]
        total_valid = int(valid_vals.size)

        if total_valid == 0:
            return WaterDetectionResult(
                water_mask=np.zeros_like(arr, dtype=bool),
                threshold_value_db=0.0,
                threshold_method="no_valid_data",
                water_pixels=0,
                total_valid_pixels=0,
                water_fraction=0.0,
                polarization_used=polarization,
                filtering_applied=filtering_applied,
                raw_threshold_db=None,
                accepted_threshold_db=None,
                threshold_adjusted=False,
                threshold_policy={"type": "none", "reason": "no_valid_data"},
                limitations=["No valid SAR data available for water detection"],
            )

        # Adaptive thresholding
        if manual_threshold_db is not None:
            raw_threshold = float(manual_threshold_db)
            accepted_threshold = raw_threshold
            threshold_adjusted = False
            threshold_policy = {
                "type": "manual_user_specified",
                "enforced": False,
                "adjusted": False,
                "value_db": accepted_threshold,
            }
            method = "manual_configured_heuristic"
        else:
            # Otsu thresholding on the dB histogram within plausible backscatter range [-35, 0] dB
            clipped = np.clip(valid_vals, -35.0, 0.0)
            hist, bin_edges = np.histogram(clipped, bins=100, range=(-35.0, 0.0))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

            # Otsu's method
            hist_norm = hist.astype(np.float64) / hist.sum()
            cum_sum = np.cumsum(hist_norm)
            cum_mean = np.cumsum(hist_norm * bin_centers)
            global_mean = cum_mean[-1]

            denom = cum_sum * (1.0 - cum_sum)
            valid_idx = (denom > 1e-7)
            if np.any(valid_idx):
                numerator = (global_mean * cum_sum - cum_mean) ** 2
                between_class_var = np.zeros_like(denom)
                between_class_var[valid_idx] = numerator[valid_idx] / denom[valid_idx]
                best_idx = np.argmax(between_class_var)
                otsu_thresh = float(bin_centers[best_idx])
                raw_threshold = float(round(otsu_thresh, 2))

                if self.enforce_sanity_bounds:
                    accepted_thresh = float(np.clip(otsu_thresh, self.sanity_min_db, self.sanity_max_db))
                    accepted_threshold = float(round(accepted_thresh, 2))
                    threshold_adjusted = bool(accepted_threshold != raw_threshold)
                    threshold_policy = {
                        "type": "configurable_sanity_bounds",
                        "min_db": self.sanity_min_db,
                        "max_db": self.sanity_max_db,
                        "enforced": True,
                        "adjusted": threshold_adjusted,
                        "notes": (
                            f"Configurable engineering sanity policy [{self.sanity_min_db}, {self.sanity_max_db}] dB applied. "
                            "This is NOT a universal physical threshold across all SAR sensors, incidence angles, and conditions."
                        ),
                    }
                    if threshold_adjusted:
                        method = f"adaptive_otsu_clamped_by_sanity_policy (raw: {raw_threshold} dB, accepted: {accepted_threshold} dB)"
                    else:
                        method = f"adaptive_otsu_within_sanity_policy (accepted: {accepted_threshold} dB)"
                else:
                    accepted_threshold = raw_threshold
                    threshold_adjusted = False
                    threshold_policy = {
                        "type": "unbounded_adaptive_otsu",
                        "enforced": False,
                        "adjusted": False,
                    }
                    method = f"unbounded_adaptive_otsu_histogram (computed {accepted_threshold} dB)"
            else:
                # Fallback: lower 15th percentile
                raw_threshold = float(round(float(np.percentile(valid_vals, 15)), 2))
                accepted_threshold = raw_threshold
                threshold_adjusted = False
                threshold_policy = {
                    "type": "distribution_percentile_fallback",
                    "percentile": 15,
                    "enforced": False,
                    "adjusted": False,
                }
                method = "distribution_15th_percentile_fallback"

        threshold = accepted_threshold

        # Water exhibits low specular backscatter
        water_mask = valid_mask & (proc_arr < threshold)
        water_count = int(np.sum(water_mask))
        water_frac = round(water_count / total_valid, 4) if total_valid > 0 else 0.0

        return WaterDetectionResult(
            water_mask=water_mask,
            threshold_value_db=round(threshold, 2),
            threshold_method=method,
            water_pixels=water_count,
            total_valid_pixels=total_valid,
            water_fraction=water_frac,
            polarization_used=polarization,
            filtering_applied=filtering_applied,
            raw_threshold_db=raw_threshold,
            accepted_threshold_db=accepted_threshold,
            threshold_adjusted=threshold_adjusted,
            threshold_policy=threshold_policy,
            limitations=[
                "Scene-dependent backscatter thresholding based on specular reflection; does NOT constitute calibrated water-detection accuracy.",
                "Sanity bounds [-25.0 dB, -12.0 dB] represent a configurable engineering policy, NOT a universal physical law across all SAR sensors, incidence angles, polarizations, or weather states.",
                "Wind-induced surface roughness can elevate water backscatter, causing false negatives.",
                "Smooth flat surfaces (tarmac, sand dunes) or radar shadow regions can cause false positive water detections.",
                "No optical or NIR confirmation available in single-image SAR mode.",
            ],
        )


class SARStructuredResponseComposer:
    """
    Composes truthful, factual natural language summaries and structured results
    from actual measured SAR radar features. NEVER invents or hallucinates values.
    """

    def compose(
        self,
        query: str,
        stats_vv: Optional[BackscatterStats],
        stats_vh: Optional[BackscatterStats],
        ratio: Optional[PolarizationRatioResult],
        water: Optional[WaterDetectionResult],
        lee_filter_info: Optional[SpeckleFilterResult],
    ) -> AnalysisResult:
        measurements: dict[str, Any] = {}
        evidence: list[str] = []
        limitations: list[str] = [
            "Deterministic radar backscatter extraction; no SAR-native VLM used on this host.",
            "Radar speckle noise mitigated via Lee filter in linear power domain.",
        ]

        summary_lines = []

        if stats_vv and stats_vv.valid_pixels > 0:
            measurements["vv_backscatter"] = {
                "mean_db": round(stats_vv.mean, 2),
                "std_db": round(stats_vv.std, 2),
                "min_db": round(stats_vv.min, 2),
                "max_db": round(stats_vv.max, 2),
                "valid_pixels": stats_vv.valid_pixels,
            }
            evidence.append(
                f"VV backscatter mean: {round(stats_vv.mean, 2)} dB (std: {round(stats_vv.std, 2)} dB) over {stats_vv.valid_pixels} pixels."
            )
            summary_lines.append(
                f"The VV co-polarization channel exhibits a mean backscatter of {round(stats_vv.mean, 2)} dB "
                f"(range: [{round(stats_vv.min, 2)}, {round(stats_vv.max, 2)}] dB)."
            )

        if stats_vh and stats_vh.valid_pixels > 0:
            measurements["vh_backscatter"] = {
                "mean_db": round(stats_vh.mean, 2),
                "std_db": round(stats_vh.std, 2),
                "min_db": round(stats_vh.min, 2),
                "max_db": round(stats_vh.max, 2),
                "valid_pixels": stats_vh.valid_pixels,
            }
            evidence.append(
                f"VH backscatter mean: {round(stats_vh.mean, 2)} dB (std: {round(stats_vh.std, 2)} dB)."
            )
            summary_lines.append(
                f"The VH cross-polarization channel shows a mean backscatter of {round(stats_vh.mean, 2)} dB."
            )

        if ratio and ratio.valid_pixels > 0:
            measurements["polarization_ratio"] = {
                "vv_vh_ratio_linear": ratio.vv_vh_ratio_linear,
                "vv_minus_vh_db": ratio.vv_minus_vh_db,
                "valid_pixels": ratio.valid_pixels,
            }
            evidence.append(
                f"Cross-polarization difference (VV - VH): {ratio.vv_minus_vh_db} dB (linear ratio: {ratio.vv_vh_ratio_linear})."
            )
            summary_lines.append(
                f"The co-to-cross polarization difference is {ratio.vv_minus_vh_db} dB (linear VV/VH ratio = {ratio.vv_vh_ratio_linear})."
            )

        if water and water.total_valid_pixels > 0:
            measurements["water_detection"] = {
                "water_fraction_pct": round(water.water_fraction * 100, 2),
                "water_pixels": water.water_pixels,
                "threshold_db": water.threshold_value_db,
                "raw_threshold_db": water.raw_threshold_db,
                "accepted_threshold_db": water.accepted_threshold_db,
                "threshold_adjusted": water.threshold_adjusted,
                "threshold_policy": water.threshold_policy,
                "threshold_method": water.threshold_method,
            }
            evidence.append(
                f"Adaptive water threshold: {water.threshold_value_db} dB ({water.threshold_method}), identified {water.water_pixels} candidate water pixels ({round(water.water_fraction * 100, 2)}%)."
            )
            summary_lines.append(
                f"Adaptive specular thresholding ({water.threshold_value_db} dB) detected candidate water over {round(water.water_fraction * 100, 2)}% of valid scene area."
            )
            limitations.extend(water.limitations)

        if lee_filter_info:
            measurements["speckle_filter"] = {
                "window_size": lee_filter_info.window_size,
                "assumed_enl": lee_filter_info.equivalent_number_of_looks,
                "filter_domain": "linear_power",
            }
            limitations.extend(lee_filter_info.assumptions)

        answer = " ".join(summary_lines) if summary_lines else "SAR analysis completed; no valid radar pixels found."

        return AnalysisResult(
            task=TaskType.SINGLE_IMAGE_VQA_SAR,
            mechanism="deterministic_sar_analysis",
            model=None,
            answer=answer,
            evidence=evidence,
            confidence={
                "confidence_type": "heuristic_uncalibrated",
                "calibrated": False,
                "score": 1.0,
                "note": "Deterministic physical radar measurements; no probabilistic hallucination.",
            },
            limitations=list(dict.fromkeys(limitations)),  # deduplicate
            status="EXECUTED",
        )


class DeterministicSAREngine:
    """
    High-level orchestrator for deterministic SAR image analysis.
    Reads raster bands, executes the full toolchain, and composes results.
    """

    def __init__(self):
        self.stats_analyzer = SARBackscatterAnalysis()
        self.speckle_filter = LeeSpeckleFilter(window_size=5, enl=4.0)
        self.ratio_estimator = PolarizationRatioEstimator()
        self.water_detector = SARWaterDetector()
        self.composer = SARStructuredResponseComposer()

    def analyze_raster(self, raster_path: str | Path, query: str = "") -> tuple[AnalysisResult, list[str]]:
        """
        Execute full SAR toolchain on a SAR GeoTIFF file.
        Returns AnalysisResult and list of executed tool names.
        """
        executed_tools = ["RasterInspector"]
        path = Path(raster_path)

        with rasterio.open(path) as ds:
            band_names = [ds.descriptions[i - 1] or f"band_{i}" for i in range(1, ds.count + 1)]
            nodata = ds.nodata

            # Identify VV and VH bands
            vv_idx = None
            vh_idx = None
            for idx, bname in enumerate(band_names):
                bn = bname.lower()
                if "vv" in bn:
                    vv_idx = idx + 1
                elif "vh" in bn:
                    vh_idx = idx + 1

            # Fallback if names are generic
            if vv_idx is None and ds.count >= 1:
                vv_idx = 1
            if vh_idx is None and ds.count >= 2:
                vh_idx = 2

            vv_arr = ds.read(vv_idx).astype(np.float64) if vv_idx else None
            vh_arr = ds.read(vh_idx).astype(np.float64) if vh_idx else None

        # 1. Backscatter Stats
        stats_vv = None
        stats_vh = None
        if vv_arr is not None:
            stats_vv = self.stats_analyzer.analyze(vv_arr, polarization="VV", input_domain="dB", nodata=nodata)
        if vh_arr is not None:
            stats_vh = self.stats_analyzer.analyze(vh_arr, polarization="VH", input_domain="dB", nodata=nodata)
        executed_tools.append("SARBackscatterAnalysis")

        # 2. Speckle Filter
        lee_info = None
        filtered_vv = vv_arr
        if vv_arr is not None:
            lee_info = self.speckle_filter.filter(vv_arr, input_is_db=True, output_as_db=True, nodata=nodata)
            filtered_vv = lee_info.filtered_array
            executed_tools.append("LeeSpeckleFilter")

        # 3. Polarization Ratio
        ratio_result = None
        if vv_arr is not None and vh_arr is not None:
            ratio_result = self.ratio_estimator.estimate(vv_arr, vh_arr, vv_is_db=True, vh_is_db=True, nodata=nodata)
            executed_tools.append("PolarizationRatioEstimator")

        # 4. Adaptive Water Detector
        water_result = None
        if filtered_vv is not None:
            water_result = self.water_detector.detect(
                filtered_vv, polarization="VV", is_db=True, nodata=nodata, apply_filter=False
            )
            executed_tools.append("SARWaterDetector")

        # 5. Response Composer
        result = self.composer.compose(
            query=query,
            stats_vv=stats_vv,
            stats_vh=stats_vh,
            ratio=ratio_result,
            water=water_result,
            lee_filter_info=lee_info,
        )
        executed_tools.append("SARStructuredResponseComposer")

        return result, executed_tools
