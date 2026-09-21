"""
SatQuery AI -- Input Gateway / Raster Inspector
Section 12: GeoTIFF parser, metadata extraction, modality detection.

THE GeoTIFF PARSER MUST BE TRULY GENERIC -- extract metadata from
the file itself, never assume band names, resolution, CRS, or
sensor-specific conventions.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.errors import RasterioIOError

from src.contracts.raster_contracts import (
    PolarizationMode,
    RasterMetadata,
    SensorModality,
)

logger = logging.getLogger("satquery.gateway")

# Supported formats (Section 12)
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg"}


class RasterInspector:
    """
    Extracts metadata from a raster file. Truly generic -- reads from
    the file itself, never assumes sensor-specific conventions.
    """

    def inspect(self, file_path: str | Path) -> RasterMetadata:
        """
        Open a raster file and extract all available metadata.
        Returns a RasterMetadata object with validation status.
        """
        file_path = Path(file_path)
        file_id = str(uuid.uuid4())[:12]
        errors: list[str] = []
        warnings: list[str] = []

        # Basic file checks
        if not file_path.exists():
            return self._rejected(file_id, file_path.name, 0,
                                  [f"File not found: {file_path}"])

        file_size = file_path.stat().st_size
        if file_size == 0:
            return self._rejected(file_id, file_path.name, 0,
                                  ["File is empty (0 bytes)"])

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return self._rejected(file_id, file_path.name, file_size,
                                  [f"Unsupported format: {ext}. "
                                   f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"])

        # Attempt rasterio open
        try:
            with rasterio.open(file_path) as ds:
                return self._extract_metadata(ds, file_id, file_path, file_size,
                                              errors, warnings)
        except RasterioIOError as e:
            return self._rejected(file_id, file_path.name, file_size,
                                  [f"Cannot open raster: {e}"])
        except Exception as e:
            return self._rejected(file_id, file_path.name, file_size,
                                  [f"Unexpected error reading raster: {e}"])

    def _extract_metadata(
        self,
        ds: rasterio.DatasetReader,
        file_id: str,
        file_path: Path,
        file_size: int,
        errors: list[str],
        warnings: list[str],
    ) -> RasterMetadata:
        """Extract all metadata from an open rasterio dataset."""

        # --- CRS ---
        crs_str = None
        crs_epsg = None
        if ds.crs:
            try:
                crs_epsg = ds.crs.to_epsg()
                crs_str = f"EPSG:{crs_epsg}" if crs_epsg else ds.crs.to_wkt()
            except Exception:
                crs_str = str(ds.crs)
        else:
            warnings.append("No CRS defined in the raster file")

        # --- Transform ---
        transform_list = None
        if ds.transform:
            t = ds.transform
            transform_list = [t.a, t.b, t.c, t.d, t.e, t.f]

        # --- Bounds ---
        bounds_dict = None
        if ds.bounds:
            b = ds.bounds
            bounds_dict = {
                "left": b.left, "bottom": b.bottom,
                "right": b.right, "top": b.top
            }

        # --- Resolution ---
        res_x = abs(ds.res[0]) if ds.res else None
        res_y = abs(ds.res[1]) if ds.res else None
        res_unit = None
        if ds.crs:
            try:
                if ds.crs.is_geographic:
                    res_unit = "degrees"
                elif ds.crs.is_projected:
                    res_unit = "meters"
            except Exception:
                pass

        # --- Band info ---
        band_names = []
        band_dtypes = []
        nodata_values = []
        for i in range(1, ds.count + 1):
            desc = ds.descriptions[i - 1] if ds.descriptions else None
            band_names.append(desc or f"band_{i}")
            band_dtypes.append(str(ds.dtypes[i - 1]))
            nodata_values.append(ds.nodata)

        # --- Sensor detection ---
        modality, detection_method = self._detect_modality(ds, band_names, file_path)
        polarization = self._detect_polarization(ds, band_names, modality)
        sensor_name = self._detect_sensor(ds, file_path)
        acq_time = self._detect_acquisition_time(ds)
        proc_level = self._detect_processing_level(ds)

        # --- Format ---
        fmt = ds.driver or "unknown"
        if fmt.lower() == "gtiff":
            fmt = "GeoTIFF"

        return RasterMetadata(
            file_id=file_id,
            filename=file_path.name,
            file_size_bytes=file_size,
            format=fmt,
            width=ds.width,
            height=ds.height,
            crs=crs_str,
            crs_epsg=crs_epsg,
            transform=transform_list,
            bounds=bounds_dict,
            resolution_x=res_x,
            resolution_y=res_y,
            resolution_unit=res_unit,
            band_count=ds.count,
            band_names=band_names,
            band_dtypes=band_dtypes,
            nodata_values=nodata_values,
            modality=modality,
            detection_method=detection_method,
            polarization=polarization,
            sensor_name=sensor_name,
            acquisition_time=acq_time,
            processing_level=proc_level,
            is_valid=len(errors) == 0,
            validation_errors=errors,
            validation_warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Sensor / Modality Detection
    # ------------------------------------------------------------------

    def _detect_modality(
        self, ds: rasterio.DatasetReader,
        band_names: list[str], file_path: Path
    ) -> SensorModality:
        """
        Detect whether the raster is optical, multispectral, or SAR.
        Uses metadata tags and band structure -- never hardcoded assumptions.
        """
        tags = self._get_all_tags(ds)
        all_text = " ".join(str(v) for v in tags.values()).lower()
        fname_lower = file_path.name.lower()
        band_text = " ".join(b.lower() for b in band_names)

        # SAR indicators
        sar_keywords = ["sar", "radar", "sentinel-1", "sentinel1", "s1",
                        "risat", "alos-palsar", "radarsat", "ers-1", "ers-2",
                        "terrasar", "cosmo-skymed", "backscatter", "sigma0",
                        "gamma0", "beta0"]
        polarization_keywords = ["vv", "vh", "hh", "hv"]

        # 1. Explicit metadata in tags
        if any(kw in all_text for kw in sar_keywords):
            return SensorModality.SAR, "explicit_metadata"
        if any(kw in band_text for kw in polarization_keywords):
            return SensorModality.SAR, "polarization_metadata"
        if any(kw in fname_lower for kw in sar_keywords):
            return SensorModality.SAR, "filename_inferred_heuristic"

        # Multispectral indicators
        optical_keywords = ["sentinel-2", "sentinel2", "s2", "landsat",
                            "cartosat", "resourcesat", "modis", "spot",
                            "worldview", "pleiades", "quickbird", "ikonos"]
        if any(kw in all_text for kw in optical_keywords):
            mod = SensorModality.MULTISPECTRAL if ds.count > 3 else SensorModality.OPTICAL
            return mod, "explicit_metadata"
        if any(kw in fname_lower for kw in optical_keywords):
            mod = SensorModality.MULTISPECTRAL if ds.count > 3 else SensorModality.OPTICAL
            return mod, "filename_inferred_heuristic"

        # Band count heuristics
        if ds.count > 4:
            return SensorModality.MULTISPECTRAL, "band_count_heuristic"
        if ds.count == 4:
            return SensorModality.MULTISPECTRAL, "band_count_heuristic"

        # Data type / range heuristic (check float with negative backscatter for SAR)
        if ds.count in (1, 2):
            try:
                sample = ds.read(1, window=rasterio.windows.Window(0, 0,
                                 min(256, ds.width), min(256, ds.height)))
                if sample.dtype in (np.float32, np.float64):
                    if np.any(sample < 0):
                        return SensorModality.SAR, "dtype_range_heuristic"
            except Exception:
                pass
            return SensorModality.UNKNOWN, "dtype_range_heuristic"

        if ds.count == 3:
            return SensorModality.OPTICAL, "band_count_heuristic"

        return SensorModality.UNKNOWN, "unresolved_heuristic"

    def _detect_polarization(
        self, ds: rasterio.DatasetReader,
        band_names: list[str], modality: SensorModality
    ) -> PolarizationMode:
        """Detect SAR polarization from band descriptions or metadata."""
        if modality not in (SensorModality.SAR, SensorModality.UNKNOWN):
            return PolarizationMode.NOT_APPLICABLE

        tags = self._get_all_tags(ds)
        all_text = " ".join(str(v) for v in tags.values()).lower()
        band_text = " ".join(b.lower() for b in band_names)
        combined = all_text + " " + band_text

        has_vv = "vv" in combined
        has_vh = "vh" in combined
        has_hh = "hh" in combined
        has_hv = "hv" in combined

        if has_vv and has_vh and has_hh and has_hv:
            return PolarizationMode.QUAD
        if has_vv and has_vh:
            return PolarizationMode.VV_VH
        if has_hh and has_hv:
            return PolarizationMode.HH_HV
        if has_vv:
            return PolarizationMode.VV
        if has_vh:
            return PolarizationMode.VH
        if has_hh:
            return PolarizationMode.HH
        if has_hv:
            return PolarizationMode.HV

        return PolarizationMode.UNKNOWN

    def _detect_sensor(self, ds: rasterio.DatasetReader,
                       file_path: Path) -> Optional[str]:
        """Attempt to identify the sensor from metadata or filename."""
        tags = self._get_all_tags(ds)
        all_text = " ".join(str(v) for v in tags.values()).lower()
        fname_lower = file_path.name.lower()
        combined = all_text + " " + fname_lower

        sensor_map = {
            "sentinel-2": "Sentinel-2",
            "sentinel2": "Sentinel-2",
            "sentinel-1": "Sentinel-1",
            "sentinel1": "Sentinel-1",
            "landsat": "Landsat",
            "cartosat": "Cartosat",
            "resourcesat": "ResourceSat",
            "risat": "RISAT",
            "modis": "MODIS",
            "worldview": "WorldView",
            "pleiades": "Pleiades",
            "spot": "SPOT",
        }
        for keyword, sensor in sensor_map.items():
            if keyword in combined:
                return sensor
        return None

    def _detect_acquisition_time(
        self, ds: rasterio.DatasetReader
    ) -> Optional[datetime]:
        """Attempt to extract acquisition time from metadata tags."""
        tags = self._get_all_tags(ds)

        time_keys = [
            "acquisition_date", "acquisitiondate", "datetime",
            "date_acquired", "scene_center_time", "product_start_time",
            "tifftag_datetime", "sensing_time", "timestamp",
        ]

        for key in time_keys:
            for tag_key, tag_val in tags.items():
                if key in tag_key.lower():
                    try:
                        # Try common date formats
                        for fmt in [
                            "%Y-%m-%dT%H:%M:%S.%fZ",
                            "%Y-%m-%dT%H:%M:%SZ",
                            "%Y-%m-%dT%H:%M:%S",
                            "%Y-%m-%d %H:%M:%S",
                            "%Y-%m-%d",
                            "%Y%m%d",
                        ]:
                            try:
                                return datetime.strptime(str(tag_val).strip(), fmt)
                            except ValueError:
                                continue
                    except Exception:
                        pass
        return None

    def _detect_processing_level(
        self, ds: rasterio.DatasetReader
    ) -> Optional[str]:
        """Attempt to detect processing level from metadata."""
        tags = self._get_all_tags(ds)
        for key, val in tags.items():
            kl = key.lower()
            if "processing_level" in kl or "product_type" in kl:
                return str(val)
        return None

    def _get_all_tags(self, ds: rasterio.DatasetReader) -> dict:
        """Collect all metadata tags from the dataset."""
        tags = {}
        try:
            tags.update(ds.tags() or {})
        except Exception:
            pass
        try:
            for ns in ds.tag_namespaces():
                tags.update(ds.tags(ns=ns) or {})
        except Exception:
            pass
        return tags

    def _rejected(
        self, file_id: str, filename: str,
        file_size: int, errors: list[str]
    ) -> RasterMetadata:
        """Create a rejected metadata response."""
        return RasterMetadata(
            file_id=file_id,
            filename=filename,
            file_size_bytes=file_size,
            width=0,
            height=0,
            band_count=0,
            is_valid=False,
            validation_errors=errors,
        )
