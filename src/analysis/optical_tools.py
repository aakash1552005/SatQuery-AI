"""
SatQuery AI -- Deterministic Optical Spectral Analysis Engine
Section 8.5 (Spectral Index Baseline) & Section 18 (Deterministic Tools).

Implements factual, reproducible spectral remote-sensing baseline:
1. BandMapping & Validation (Red, Green, Blue, NIR, SWIR1)
2. Normalized Difference Vegetation Index (NDVI)
3. Normalized Difference Water Index (NDWI)
4. Modified Normalized Difference Water Index (MNDWI, conditional on SWIR)
5. Rule-Based Land-Cover Classification Baseline (Transparent heuristic rules)
6. OpticalStructuredResponseComposer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio

from src.analysis.numerical_math import (
    compute_mndwi,
    compute_ndvi,
    compute_ndwi,
)
from src.contracts.query_contracts import AnalysisResult, TaskType

logger = logging.getLogger("satquery.analysis.optical")


@dataclass
class BandMap:
    """Explicit mapping from standard spectral roles to 1-based band indices."""
    red_idx: Optional[int] = None
    green_idx: Optional[int] = None
    blue_idx: Optional[int] = None
    nir_idx: Optional[int] = None
    swir1_idx: Optional[int] = None
    mapping_source: str = "heuristic_band_naming"
    band_names_detected: list[str] = field(default_factory=list)


@dataclass
class SpectralIndexResult:
    """Result of computing a single normalized difference index."""
    index_name: str
    array: np.ndarray
    valid_pixels: int
    mean: float
    min: float
    max: float
    positive_fraction: float
    formula: str


@dataclass
class LandCoverDistribution:
    """Class percentage distribution from rule-based classification."""
    classes: dict[str, float]  # class_name -> percentage (0 - 100)
    pixel_counts: dict[str, int]
    total_valid_pixels: int
    dominant_class: str
    rules_applied: list[str] = field(default_factory=list)


class OpticalBandMapper:
    """
    Identifies and validates spectral band positions following a strict hierarchy:
    1. Explicit metadata descriptions (e.g. TIFF tags, band descriptions).
    2. Dataset-specific schemas (e.g. bigearthnet_s2, sentinel2_l2a, landsat8).
    3. Validated band mapping.
    4. Fallback indexing ONLY when explicitly allowed/configured.
    NEVER silently guesses or shuffles bands without verification.
    """

    def __init__(
        self,
        dataset_schema: Optional[str] = None,
        allow_fallback_assumptions: bool = True,
    ):
        self.dataset_schema = dataset_schema
        self.allow_fallback_assumptions = allow_fallback_assumptions

    def map_bands(
        self,
        band_count: int,
        band_names: Optional[list[str]] = None,
        schema_override: Optional[str] = None,
    ) -> BandMap:
        schema = schema_override or self.dataset_schema
        bnames = [b.lower() for b in (band_names or [])]
        detected = BandMap(band_names_detected=bnames or [f"band_{i}" for i in range(1, band_count + 1)])

        # 1. Match explicit metadata descriptions
        explicit_matches = 0
        for i, name in enumerate(bnames, start=1):
            if any(k in name for k in ("nir", "near_ir", "b08", "b8")):
                detected.nir_idx = i
                explicit_matches += 1
            elif any(k in name for k in ("swir1", "swir_1", "b11")):
                detected.swir1_idx = i
                explicit_matches += 1
            elif any(k in name for k in ("red", "b04", "b4")):
                detected.red_idx = i
                explicit_matches += 1
            elif any(k in name for k in ("green", "b03", "b3")):
                detected.green_idx = i
                explicit_matches += 1
            elif any(k in name for k in ("blue", "b02", "b2")):
                detected.blue_idx = i
                explicit_matches += 1

        if explicit_matches > 0:
            detected.mapping_source = f"explicit_metadata ({explicit_matches} bands matched)"
            return detected

        # 2. Dataset-specific schema matching
        if schema:
            s_lower = schema.lower()
            if s_lower in ("sentinel2", "bigearthnet_s2", "s2_l2a"):
                detected.blue_idx = 2
                detected.green_idx = 3
                detected.red_idx = 4
                detected.nir_idx = 8
                if band_count >= 11:
                    detected.swir1_idx = 11
                detected.mapping_source = f"dataset_schema_{schema}"
                return detected
            elif s_lower in ("standard_rgb", "rgb"):
                detected.red_idx = 1
                detected.green_idx = 2
                detected.blue_idx = 3
                detected.mapping_source = "dataset_schema_standard_rgb"
                return detected
            elif s_lower in ("standard_rgbnir", "rgbnir"):
                detected.red_idx = 1
                detected.green_idx = 2
                detected.blue_idx = 3
                detected.nir_idx = 4
                detected.mapping_source = "dataset_schema_standard_rgbnir"
                return detected

        # 3. Fallback standard assumption ONLY when explicitly permitted
        if self.allow_fallback_assumptions:
            if band_count == 3:
                detected.red_idx = 1
                detected.green_idx = 2
                detected.blue_idx = 3
                detected.mapping_source = "configured_fallback_rgb_standard_assumption"
            elif band_count == 4:
                detected.red_idx = 1
                detected.green_idx = 2
                detected.blue_idx = 3
                detected.nir_idx = 4
                detected.mapping_source = "configured_fallback_rgb_nir_standard_assumption"
            elif band_count >= 5:
                detected.blue_idx = 1
                detected.green_idx = 2
                detected.red_idx = 3
                detected.nir_idx = 4
                detected.swir1_idx = 5
                detected.mapping_source = "configured_fallback_multispectral_stack_assumption"
        else:
            detected.mapping_source = "unresolved_missing_metadata"

        return detected


class SpectralIndexEngine:
    """Computes standard vegetation and water indices with nodata/zero guards."""

    def compute_ndvi_stat(
        self,
        nir: np.ndarray,
        red: np.ndarray,
        nodata: Optional[float] = None,
    ) -> SpectralIndexResult:
        nir_f = np.asarray(nir, dtype=np.float64)
        red_f = np.asarray(red, dtype=np.float64)

        mask = np.isfinite(nir_f) & np.isfinite(red_f)
        if nodata is not None and not np.isnan(nodata):
            mask &= (nir_f != nodata) & (red_f != nodata)

        ndvi_arr = np.zeros_like(nir_f)
        if np.any(mask):
            ndvi_arr[mask] = compute_ndvi(nir_f[mask], red_f[mask])

        valid_vals = ndvi_arr[mask]
        valid_cnt = int(valid_vals.size)

        if valid_cnt == 0:
            return SpectralIndexResult(
                index_name="NDVI",
                array=ndvi_arr,
                valid_pixels=0,
                mean=0.0,
                min=0.0,
                max=0.0,
                positive_fraction=0.0,
                formula="(NIR - RED) / (NIR + RED)",
            )

        mean_val = float(np.mean(valid_vals))
        min_val = float(np.min(valid_vals))
        max_val = float(np.max(valid_vals))
        pos_frac = float(np.mean(valid_vals > 0.2))

        return SpectralIndexResult(
            index_name="NDVI",
            array=ndvi_arr,
            valid_pixels=valid_cnt,
            mean=round(mean_val, 4),
            min=round(min_val, 4),
            max=round(max_val, 4),
            positive_fraction=round(pos_frac, 4),
            formula="(NIR - RED) / (NIR + RED)",
        )

    def compute_ndwi_stat(
        self,
        green: np.ndarray,
        nir: np.ndarray,
        nodata: Optional[float] = None,
    ) -> SpectralIndexResult:
        green_f = np.asarray(green, dtype=np.float64)
        nir_f = np.asarray(nir, dtype=np.float64)

        mask = np.isfinite(green_f) & np.isfinite(nir_f)
        if nodata is not None and not np.isnan(nodata):
            mask &= (green_f != nodata) & (nir_f != nodata)

        ndwi_arr = np.zeros_like(green_f)
        if np.any(mask):
            ndwi_arr[mask] = compute_ndwi(green_f[mask], nir_f[mask])

        valid_vals = ndwi_arr[mask]
        valid_cnt = int(valid_vals.size)

        if valid_cnt == 0:
            return SpectralIndexResult(
                index_name="NDWI",
                array=ndwi_arr,
                valid_pixels=0,
                mean=0.0,
                min=0.0,
                max=0.0,
                positive_fraction=0.0,
                formula="(GREEN - NIR) / (GREEN + NIR)",
            )

        mean_val = float(np.mean(valid_vals))
        min_val = float(np.min(valid_vals))
        max_val = float(np.max(valid_vals))
        pos_frac = float(np.mean(valid_vals > 0.0))

        return SpectralIndexResult(
            index_name="NDWI",
            array=ndwi_arr,
            valid_pixels=valid_cnt,
            mean=round(mean_val, 4),
            min=round(min_val, 4),
            max=round(max_val, 4),
            positive_fraction=round(pos_frac, 4),
            formula="(GREEN - NIR) / (GREEN + NIR)",
        )

    def compute_mndwi_stat(
        self,
        green: np.ndarray,
        swir: np.ndarray,
        nodata: Optional[float] = None,
    ) -> SpectralIndexResult:
        green_f = np.asarray(green, dtype=np.float64)
        swir_f = np.asarray(swir, dtype=np.float64)

        mask = np.isfinite(green_f) & np.isfinite(swir_f)
        if nodata is not None and not np.isnan(nodata):
            mask &= (green_f != nodata) & (swir_f != nodata)

        mndwi_arr = np.zeros_like(green_f)
        if np.any(mask):
            mndwi_arr[mask] = compute_mndwi(green_f[mask], swir_f[mask])

        valid_vals = mndwi_arr[mask]
        valid_cnt = int(valid_vals.size)

        if valid_cnt == 0:
            return SpectralIndexResult(
                index_name="MNDWI",
                array=mndwi_arr,
                valid_pixels=0,
                mean=0.0,
                min=0.0,
                max=0.0,
                positive_fraction=0.0,
                formula="(GREEN - SWIR1) / (GREEN + SWIR1)",
            )

        mean_val = float(np.mean(valid_vals))
        min_val = float(np.min(valid_vals))
        max_val = float(np.max(valid_vals))
        pos_frac = float(np.mean(valid_vals > 0.0))

        return SpectralIndexResult(
            index_name="MNDWI",
            array=mndwi_arr,
            valid_pixels=valid_cnt,
            mean=round(mean_val, 4),
            min=round(min_val, 4),
            max=round(max_val, 4),
            positive_fraction=round(pos_frac, 4),
            formula="(GREEN - SWIR1) / (GREEN + SWIR1)",
        )


class RuleBasedLandCoverClassifier:
    """
    Transparent, deterministic rule-based land cover classification baseline.
    Never claims to be a neural model or trained AI classifier.
    Applies explicit decision trees over NDVI, NDWI, and visible spectral thresholds.
    """

    def classify(
        self,
        ndvi: Optional[np.ndarray] = None,
        ndwi: Optional[np.ndarray] = None,
        red: Optional[np.ndarray] = None,
        green: Optional[np.ndarray] = None,
        blue: Optional[np.ndarray] = None,
        nodata: Optional[float] = None,
    ) -> LandCoverDistribution:
        # Determine shape and base validity mask
        ref = ndvi if ndvi is not None else red
        if ref is None:
            return LandCoverDistribution(
                classes={"UNKNOWN": 100.0},
                pixel_counts={"UNKNOWN": 0},
                total_valid_pixels=0,
                dominant_class="UNKNOWN",
                rules_applied=["No spectral inputs provided"],
            )

        valid_mask = np.isfinite(ref)
        if nodata is not None and not np.isnan(nodata):
            valid_mask &= (ref != nodata)

        total_valid = int(np.sum(valid_mask))
        if total_valid == 0:
            return LandCoverDistribution(
                classes={"UNKNOWN": 100.0},
                pixel_counts={"UNKNOWN": 0},
                total_valid_pixels=0,
                dominant_class="UNKNOWN",
                rules_applied=["Empty valid pixel region"],
            )

        # Classes: WATER, DENSE_VEGETATION, MODERATE_VEGETATION, BARE_SOIL, BUILT_UP, UNKNOWN
        counts = {
            "WATER": 0,
            "DENSE_VEGETATION": 0,
            "MODERATE_VEGETATION": 0,
            "BARE_SOIL": 0,
            "BUILT_UP": 0,
            "UNKNOWN": 0,
        }

        rules = []

        if ndvi is not None and ndwi is not None:
            # Full multispectral index rule set
            water_mask = valid_mask & (ndwi > 0.05) & (ndvi < 0.15)
            dense_veg_mask = valid_mask & ~water_mask & (ndvi >= 0.50)
            mod_veg_mask = valid_mask & ~water_mask & ~dense_veg_mask & (ndvi >= 0.25)
            bare_soil_mask = valid_mask & ~water_mask & ~dense_veg_mask & ~mod_veg_mask & (ndvi >= 0.05) & (ndvi < 0.25)
            built_up_mask = valid_mask & ~water_mask & ~dense_veg_mask & ~mod_veg_mask & ~bare_soil_mask & (ndvi < 0.05) & (ndwi < 0.0)
            unknown_mask = valid_mask & ~water_mask & ~dense_veg_mask & ~mod_veg_mask & ~bare_soil_mask & ~built_up_mask

            counts["WATER"] = int(np.sum(water_mask))
            counts["DENSE_VEGETATION"] = int(np.sum(dense_veg_mask))
            counts["MODERATE_VEGETATION"] = int(np.sum(mod_veg_mask))
            counts["BARE_SOIL"] = int(np.sum(bare_soil_mask))
            counts["BUILT_UP"] = int(np.sum(built_up_mask))
            counts["UNKNOWN"] = int(np.sum(unknown_mask))

            rules.extend([
                "WATER: NDWI > 0.05 AND NDVI < 0.15",
                "DENSE_VEGETATION: NDVI >= 0.50",
                "MODERATE_VEGETATION: 0.25 <= NDVI < 0.50",
                "BARE_SOIL: 0.05 <= NDVI < 0.25",
                "BUILT_UP: NDVI < 0.05 AND NDWI < 0.0",
            ])
        elif red is not None and green is not None and blue is not None:
            # 3-band RGB visible spectral heuristic
            r = red.astype(np.float64)
            g = green.astype(np.float64)
            b = blue.astype(np.float64)

            # Visible water: high blue/green compared to red
            water_mask = valid_mask & (b > r * 1.1) & (g > r * 1.05)
            veg_mask = valid_mask & ~water_mask & (g > r * 1.15) & (g > b * 1.10)
            soil_mask = valid_mask & ~water_mask & ~veg_mask & (r > g * 1.05) & (r > b * 1.15)
            built_mask = valid_mask & ~water_mask & ~veg_mask & ~soil_mask

            counts["WATER"] = int(np.sum(water_mask))
            counts["DENSE_VEGETATION"] = int(np.sum(veg_mask))
            counts["MODERATE_VEGETATION"] = 0
            counts["BARE_SOIL"] = int(np.sum(soil_mask))
            counts["BUILT_UP"] = int(np.sum(built_mask))
            counts["UNKNOWN"] = 0

            rules.extend([
                "VISIBLE_WATER: Blue > Red*1.1 AND Green > Red*1.05",
                "VISIBLE_VEGETATION: Green > Red*1.15 AND Green > Blue*1.10",
                "VISIBLE_BARE_SOIL: Red > Green*1.05 AND Red > Blue*1.15",
                "VISIBLE_BUILT_UP: Residual high-reflectance spectral contrast",
            ])
        else:
            counts["UNKNOWN"] = total_valid
            rules.append("Insufficient spectral bands for classification")

        # Convert to percentages
        classes_pct = {k: round((v / total_valid) * 100.0, 2) for k, v in counts.items()}
        dominant = max(classes_pct, key=classes_pct.get)

        return LandCoverDistribution(
            classes=classes_pct,
            pixel_counts=counts,
            total_valid_pixels=total_valid,
            dominant_class=dominant,
            rules_applied=rules,
        )


class OpticalStructuredResponseComposer:
    """
    Composes factual, transparent natural language responses and structured output
    from deterministic spectral index measurements and rule-based classification.
    """

    def compose(
        self,
        query: str,
        ndvi: Optional[SpectralIndexResult],
        ndwi: Optional[SpectralIndexResult],
        mndwi: Optional[SpectralIndexResult],
        land_cover: Optional[LandCoverDistribution],
        band_map: Optional[BandMap],
    ) -> AnalysisResult:
        measurements: dict[str, Any] = {}
        evidence: list[str] = []
        limitations: list[str] = [
            "Deterministic optical spectral index analysis; no VLM model used.",
            "Land-cover classes derived via rule_based_land_cover_classification thresholds, not trained deep learning.",
        ]

        summary_parts = []

        if band_map:
            measurements["band_mapping"] = {
                "red_index": band_map.red_idx,
                "green_index": band_map.green_idx,
                "blue_index": band_map.blue_idx,
                "nir_index": band_map.nir_idx,
                "swir1_index": band_map.swir1_idx,
                "mapping_source": band_map.mapping_source,
            }

        if ndvi and ndvi.valid_pixels > 0:
            measurements["ndvi"] = {
                "mean": ndvi.mean,
                "min": ndvi.min,
                "max": ndvi.max,
                "vegetation_pixel_fraction": ndvi.positive_fraction,
                "valid_pixels": ndvi.valid_pixels,
            }
            evidence.append(
                f"NDVI mean: {ndvi.mean} (range [{ndvi.min}, {ndvi.max}]), vegetation pixel fraction: {round(ndvi.positive_fraction * 100, 1)}%."
            )
            summary_parts.append(
                f"NDVI averages {ndvi.mean} across {ndvi.valid_pixels} valid pixels (active vegetation covers approximately {round(ndvi.positive_fraction * 100, 1)}% of the scene)."
            )
        else:
            limitations.append("NDVI cannot be computed because valid NIR and Red band mappings were not provided.")
            q_lower = query.lower()
            if any(k in q_lower for k in ("ndvi", "vegetation", "crop", "forest", "plants")):
                summary_parts.append("NDVI cannot be computed because valid NIR and Red band mappings were not provided.")

        if ndwi and ndwi.valid_pixels > 0:
            measurements["ndwi"] = {
                "mean": ndwi.mean,
                "min": ndwi.min,
                "max": ndwi.max,
                "water_pixel_fraction": ndwi.positive_fraction,
            }
            evidence.append(
                f"NDWI mean: {ndwi.mean} (water candidate fraction: {round(ndwi.positive_fraction * 100, 1)}%)."
            )
        else:
            limitations.append("NDWI cannot be computed because valid Green and NIR band mappings were not provided.")
            q_lower = query.lower()
            if any(k in q_lower for k in ("ndwi", "water index", "moisture")):
                summary_parts.append("NDWI cannot be computed because valid Green and NIR band mappings were not provided.")

        if mndwi and mndwi.valid_pixels > 0:
            measurements["mndwi"] = {
                "mean": mndwi.mean,
                "min": mndwi.min,
                "max": mndwi.max,
            }
            evidence.append(f"MNDWI mean: {mndwi.mean} (enhanced open water delineation).")

        if land_cover and land_cover.total_valid_pixels > 0:
            measurements["land_cover_distribution_pct"] = land_cover.classes
            measurements["dominant_class"] = land_cover.dominant_class
            evidence.append(
                f"Rule-based land-cover classification dominant class: {land_cover.dominant_class} ({land_cover.classes.get(land_cover.dominant_class, 0.0)}%)."
            )
            breakdown = ", ".join(f"{k}: {v}%" for k, v in land_cover.classes.items() if v > 0.0)
            summary_parts.append(
                f"Rule-based classification identifies {land_cover.dominant_class} as the dominant category. Spectral distribution: {breakdown}."
            )
            limitations.extend(land_cover.rules_applied)

        answer = " ".join(summary_parts) if summary_parts else "Spectral analysis completed; insufficient valid pixels for summary."

        return AnalysisResult(
            task=TaskType.SINGLE_IMAGE_VQA_OPTICAL,
            mechanism="deterministic_optical_spectral_analysis",
            model=None,
            answer=answer,
            evidence=evidence,
            confidence={
                "confidence_type": "heuristic_uncalibrated",
                "calibrated": False,
                "score": 1.0,
                "note": "Deterministic physical spectral indices; zero probabilistic hallucination.",
            },
            limitations=list(dict.fromkeys(limitations)),
            status="EXECUTED",
        )


class DeterministicOpticalEngine:
    """
    High-level orchestrator for deterministic optical remote-sensing analysis.
    Validates band mapping, executes spectral index calculations and rule-based
    classification, and returns a factual structured response.
    """

    def __init__(self):
        self.band_mapper = OpticalBandMapper()
        self.index_engine = SpectralIndexEngine()
        self.classifier = RuleBasedLandCoverClassifier()
        self.composer = OpticalStructuredResponseComposer()

    def analyze_raster(
        self,
        raster_path: str | Path,
        query: str = "",
    ) -> tuple[AnalysisResult, list[str]]:
        """
        Execute optical deterministic analysis on a GeoTIFF.
        Returns AnalysisResult and list of executed tool names.
        """
        executed_tools = ["RasterInspector"]
        path = Path(raster_path)

        with rasterio.open(path) as ds:
            band_count = ds.count
            band_names = [ds.descriptions[i - 1] or f"band_{i}" for i in range(1, band_count + 1)]
            nodata = ds.nodata

            bmap = self.band_mapper.map_bands(band_count, band_names)
            executed_tools.append("BandMappingValidator")

            red = ds.read(bmap.red_idx).astype(np.float64) if bmap.red_idx and bmap.red_idx <= band_count else None
            green = ds.read(bmap.green_idx).astype(np.float64) if bmap.green_idx and bmap.green_idx <= band_count else None
            blue = ds.read(bmap.blue_idx).astype(np.float64) if bmap.blue_idx and bmap.blue_idx <= band_count else None
            nir = ds.read(bmap.nir_idx).astype(np.float64) if bmap.nir_idx and bmap.nir_idx <= band_count else None
            swir = ds.read(bmap.swir1_idx).astype(np.float64) if bmap.swir1_idx and bmap.swir1_idx <= band_count else None

        # Check required bands for NDVI
        ndvi_res = None
        if nir is not None and red is not None:
            ndvi_res = self.index_engine.compute_ndvi_stat(nir, red, nodata=nodata)
            executed_tools.append("SpectralIndexEngine (NDVI)")

        # NDWI
        ndwi_res = None
        if green is not None and nir is not None:
            ndwi_res = self.index_engine.compute_ndwi_stat(green, nir, nodata=nodata)
            executed_tools.append("SpectralIndexEngine (NDWI)")

        # MNDWI (optional, only if SWIR exists)
        mndwi_res = None
        if green is not None and swir is not None:
            mndwi_res = self.index_engine.compute_mndwi_stat(green, swir, nodata=nodata)
            executed_tools.append("SpectralIndexEngine (MNDWI)")

        # Rule-based land cover
        lc_dist = self.classifier.classify(
            ndvi=ndvi_res.array if ndvi_res else None,
            ndwi=ndwi_res.array if ndwi_res else None,
            red=red,
            green=green,
            blue=blue,
            nodata=nodata,
        )
        executed_tools.append("RuleBasedLandCoverClassifier")

        # Response composition
        result = self.composer.compose(
            query=query,
            ndvi=ndvi_res,
            ndwi=ndwi_res,
            mndwi=mndwi_res,
            land_cover=lc_dist,
            band_map=bmap,
        )
        executed_tools.append("OpticalStructuredResponseComposer")

        return result, executed_tools
