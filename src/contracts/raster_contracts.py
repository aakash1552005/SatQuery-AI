"""
SatQuery AI -- Pydantic contracts for raster metadata and data readiness.
Section 12 (Input Gateway) + Section 13 (Data Readiness Gate).
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SensorModality(str, enum.Enum):
    """Detected sensor modality of a raster input."""
    OPTICAL = "optical"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    UNKNOWN = "unknown"


class PolarizationMode(str, enum.Enum):
    """SAR polarization mode."""
    VV = "vv"
    VH = "vh"
    VV_VH = "vv_vh"
    HH = "hh"
    HV = "hv"
    HH_HV = "hh_hv"
    QUAD = "quad"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class RasterMetadata(BaseModel):
    """
    Metadata extracted from a single raster file.
    Section 12: dimensions, CRS, transform, bounds, resolution,
    band count, band metadata, nodata, acquisition time, modality,
    sensor metadata, polarization.
    """
    file_id: str = Field(..., description="Unique identifier for this upload")
    filename: str
    file_size_bytes: int
    format: str = Field(default="unknown", description="e.g. GeoTIFF, TIFF, PNG, JPEG")

    # Spatial
    width: int
    height: int
    crs: Optional[str] = Field(None, description="CRS as EPSG string or WKT")
    crs_epsg: Optional[int] = None
    transform: Optional[list[float]] = Field(None, description="Affine transform as 6 floats")
    bounds: Optional[dict] = Field(None, description="left, bottom, right, top")
    resolution_x: Optional[float] = Field(None, description="Pixel size in X (map units)")
    resolution_y: Optional[float] = Field(None, description="Pixel size in Y (map units)")
    resolution_unit: Optional[str] = Field(None, description="e.g. 'meters', 'degrees'")

    # Spectral
    band_count: int
    band_names: Optional[list[str]] = None
    band_dtypes: Optional[list[str]] = None
    nodata_values: Optional[list] = None

    # Sensor
    modality: SensorModality = SensorModality.UNKNOWN
    polarization: PolarizationMode = PolarizationMode.NOT_APPLICABLE
    sensor_name: Optional[str] = None
    acquisition_time: Optional[datetime] = None
    processing_level: Optional[str] = None

    # File integrity
    is_valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)


class CompatibilityStatus(str, enum.Enum):
    """Result of compatibility checking for pairs."""
    COMPATIBLE = "compatible"
    COMPATIBLE_WITH_WARNINGS = "compatible_with_warnings"
    INCOMPATIBLE = "incompatible"


class PairCompatibility(BaseModel):
    """Compatibility check result for a raster pair (temporal or optical-SAR)."""
    status: CompatibilityStatus
    pair_type: str = Field(..., description="temporal, optical_sar, or unknown")
    crs_match: bool
    bounds_overlap: Optional[float] = Field(None, description="Fraction 0-1 of spatial overlap")
    resolution_match: bool
    resolution_ratio: Optional[float] = None
    temporal_gap_days: Optional[float] = None
    modality_pair: Optional[str] = Field(None, description="e.g. optical-optical, optical-sar")
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_compatible(self) -> bool:
        return self.status in (CompatibilityStatus.COMPATIBLE, CompatibilityStatus.COMPATIBLE_WITH_WARNINGS)


class UploadResponse(BaseModel):
    """API response for a file upload."""
    file_id: str
    filename: str
    metadata: RasterMetadata
    status: str = Field(default="accepted", description="accepted, rejected, or error")
    rejection_reason: Optional[str] = None
