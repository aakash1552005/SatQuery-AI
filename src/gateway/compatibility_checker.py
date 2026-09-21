"""
SatQuery AI -- Data Readiness Gate / Compatibility Checker
Section 13: CRS, bounds, overlap, dimensions, resolution, temporal, modality.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.contracts.raster_contracts import (
    CompatibilityStatus,
    PairCompatibility,
    RasterMetadata,
    SensorModality,
)

logger = logging.getLogger("satquery.gateway.compatibility")


class CompatibilityChecker:
    """
    Checks whether raster pairs are compatible for temporal change
    detection or optical-SAR analysis.

    Note on Spatial Compatibility:
    CRS equality does not by itself guarantee spatial compatibility.
    Compatibility requires CRS alignment, spatial footprint overlap,
    comparable Ground Sample Distance (GSD), grid alignment, and
    compatible sensor modalities.

    The GSD threshold is an engineering policy threshold, not an
    absolute physical or universal scientific law.
    """

    def __init__(self, max_gsd_ratio_policy: float = 3.0):
        self.max_gsd_ratio_policy = max_gsd_ratio_policy

    def check_pair(
        self,
        meta_a: RasterMetadata,
        meta_b: RasterMetadata,
    ) -> PairCompatibility:
        """Check compatibility of two rasters as a pair."""
        issues: list[str] = []
        warnings: list[str] = []

        # --- Determine pair type ---
        pair_type = self._determine_pair_type(meta_a, meta_b)

        # --- CRS check ---
        crs_match = self._check_crs(meta_a, meta_b, issues, warnings)

        # --- Bounds overlap ---
        overlap = self._check_overlap(meta_a, meta_b, issues, warnings)

        # --- Resolution ---
        res_match, res_ratio = self._check_resolution(
            meta_a, meta_b, issues, warnings
        )

        # --- Temporal gap (for temporal pairs) ---
        temporal_gap = self._check_temporal(
            meta_a, meta_b, pair_type, issues, warnings
        )

        # --- Modality ---
        modality_pair = f"{meta_a.modality.value}-{meta_b.modality.value}"

        # --- Status ---
        if issues:
            status = CompatibilityStatus.INCOMPATIBLE
        elif warnings:
            status = CompatibilityStatus.COMPATIBLE_WITH_WARNINGS
        else:
            status = CompatibilityStatus.COMPATIBLE

        return PairCompatibility(
            status=status,
            pair_type=pair_type,
            crs_match=crs_match,
            bounds_overlap=overlap,
            resolution_match=res_match,
            resolution_ratio=res_ratio,
            temporal_gap_days=temporal_gap,
            modality_pair=modality_pair,
            issues=issues,
            warnings=warnings,
        )

    def _determine_pair_type(
        self, a: RasterMetadata, b: RasterMetadata
    ) -> str:
        """Determine whether the pair is temporal, optical-SAR, or unknown."""
        a_mod = a.modality
        b_mod = b.modality

        # Optical-SAR pair
        if (a_mod in (SensorModality.OPTICAL, SensorModality.MULTISPECTRAL)
                and b_mod == SensorModality.SAR):
            return "optical_sar"
        if (b_mod in (SensorModality.OPTICAL, SensorModality.MULTISPECTRAL)
                and a_mod == SensorModality.SAR):
            return "optical_sar"

        # Temporal pair (same modality, different times)
        if a.acquisition_time and b.acquisition_time:
            if a.acquisition_time != b.acquisition_time:
                return "temporal"

        # Same modality but no timestamps -- could be temporal
        if a_mod == b_mod:
            return "temporal"

        return "unknown"

    def _check_crs(
        self, a: RasterMetadata, b: RasterMetadata,
        issues: list[str], warnings: list[str]
    ) -> bool:
        """Check CRS compatibility."""
        if not a.crs and not b.crs:
            warnings.append("Neither raster has CRS defined")
            return False
        if not a.crs or not b.crs:
            warnings.append(
                f"CRS missing: {a.filename if not a.crs else b.filename}"
            )
            return False

        if a.crs_epsg and b.crs_epsg:
            match = a.crs_epsg == b.crs_epsg
        else:
            match = a.crs == b.crs

        if not match:
            warnings.append(
                f"CRS mismatch: {a.crs} vs {b.crs}. "
                f"Reprojection will be required."
            )
        return match

    def _check_overlap(
        self, a: RasterMetadata, b: RasterMetadata,
        issues: list[str], warnings: list[str]
    ) -> Optional[float]:
        """Check spatial overlap between two rasters."""
        if not a.bounds or not b.bounds:
            warnings.append("Cannot compute spatial overlap (missing bounds)")
            return None

        ab = a.bounds
        bb = b.bounds

        # Compute intersection
        x_left = max(ab["left"], bb["left"])
        x_right = min(ab["right"], bb["right"])
        y_bottom = max(ab["bottom"], bb["bottom"])
        y_top = min(ab["top"], bb["top"])

        if x_left >= x_right or y_bottom >= y_top:
            issues.append("No spatial overlap between the two rasters")
            return 0.0

        intersect_area = (x_right - x_left) * (y_top - y_bottom)
        a_area = (ab["right"] - ab["left"]) * (ab["top"] - ab["bottom"])
        b_area = (bb["right"] - bb["left"]) * (bb["top"] - bb["bottom"])
        min_area = min(a_area, b_area)

        if min_area <= 0:
            warnings.append("Degenerate bounds (zero area)")
            return None

        overlap_fraction = round(intersect_area / min_area, 4)

        if overlap_fraction < 0.1:
            issues.append(
                f"Spatial overlap too small: {overlap_fraction:.1%}"
            )
        elif overlap_fraction < 0.5:
            warnings.append(
                f"Limited spatial overlap: {overlap_fraction:.1%}"
            )

        return overlap_fraction

    def _check_resolution(
        self, a: RasterMetadata, b: RasterMetadata,
        issues: list[str], warnings: list[str]
    ) -> tuple[bool, Optional[float]]:
        """Check resolution compatibility."""
        if not a.resolution_x or not b.resolution_x:
            warnings.append("Cannot compare resolution (missing metadata)")
            return False, None

        ratio = max(a.resolution_x, b.resolution_x) / min(
            a.resolution_x, b.resolution_x
        )

        if ratio > 10:
            warnings.append(
                f"Large resolution mismatch: {a.resolution_x} vs "
                f"{b.resolution_x} (ratio: {ratio:.1f}x)"
            )
            return False, round(ratio, 2)
        elif ratio > 2:
            warnings.append(
                f"Resolution difference: {a.resolution_x} vs "
                f"{b.resolution_x} (ratio: {ratio:.1f}x)"
            )
            return False, round(ratio, 2)

        return ratio < 1.5, round(ratio, 2)

    def _check_temporal(
        self, a: RasterMetadata, b: RasterMetadata,
        pair_type: str, issues: list[str], warnings: list[str]
    ) -> Optional[float]:
        """Check temporal gap for temporal pairs."""
        if pair_type not in ("temporal",):
            return None

        if not a.acquisition_time or not b.acquisition_time:
            warnings.append(
                "Temporal analysis requested but acquisition time "
                "missing from one or both rasters"
            )
            return None

        delta = abs(
            (b.acquisition_time - a.acquisition_time).total_seconds()
        )
        days = delta / 86400.0

        if days < 0.01:
            issues.append(
                "Acquisition times are identical or near-identical -- "
                "no temporal change expected"
            )
        elif days > 3650:  # ~10 years
            warnings.append(
                f"Very large temporal gap: {days:.0f} days ({days/365:.1f} years)"
            )

        return round(days, 2)
