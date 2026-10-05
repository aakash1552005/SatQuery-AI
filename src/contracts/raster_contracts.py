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


class InputSource(str, enum.Enum):
    """
    Explicit dataset provenance and origin for raster inputs.
    Enforces strict separation between user uploads, synthetic engineering validation,
    adaptation datasets (BigEarthNet), and evaluation benchmarks (VRSBench).
    """
    USER_UPLOAD = "USER_UPLOAD"
    SYNTHETIC_ENGINEERING = "SYNTHETIC_ENGINEERING"
    BIGEARTHNET_TXT = "BIGEARTHNET_TXT"
    VRSBENCH = "VRSBENCH"
    UNKNOWN = "UNKNOWN"


class DatasetRole(str, enum.Enum):
    """
    Operational role of dataset.
    Enforces governance rules:
    - Training datasets must never automatically become evaluation datasets.
    - VRSBench evaluation data must never be used for fine-tuning.
    """
    INFERENCE = "INFERENCE"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    BENCHMARK_EVALUATION = "BENCHMARK_EVALUATION"
    UNASSIGNED = "UNASSIGNED"


def validate_dataset_governance(source: InputSource, role: DatasetRole) -> tuple[bool, Optional[str]]:
    """
    Validate dataset governance and integrity rules:
    1. Training datasets must never automatically become evaluation datasets.
    2. VRSBench evaluation data must never be used for fine-tuning/training.
    """
    if source == InputSource.VRSBENCH and role in (DatasetRole.TRAINING, DatasetRole.VALIDATION):
        return False, "Governance Violation: VRSBench evaluation data must never be used for fine-tuning or training."
    if source == InputSource.BIGEARTHNET_TXT and role == DatasetRole.BENCHMARK_EVALUATION:
        return False, "Governance Violation: Training/adaptation datasets (BigEarthNet) must never automatically become benchmark evaluation datasets."
    return True, None


class SpatialBounds(BaseModel):
    """Spatial bounding box coordinates in native or geographic CRS."""
    left: float
    bottom: float
    right: float
    top: float


class RasterMetadata(BaseModel):
    """
    Metadata extracted from a single raster file.
    Section 12: dimensions, CRS, transform, bounds, resolution,
    band count, band metadata, nodata, acquisition time, modality,
    sensor metadata, polarization, input_source, dataset_role.
    """
    file_id: str = Field(default="file_default", description="Unique identifier for this upload")
    filename: str
    file_size_bytes: int = Field(default=0, description="File size in bytes")
    format: str = Field(default="unknown", description="e.g. GeoTIFF, TIFF, PNG, JPEG")
    input_source: InputSource = Field(
        default=InputSource.USER_UPLOAD,
        description="Dataset provenance: USER_UPLOAD, SYNTHETIC_ENGINEERING, BIGEARTHNET_TXT, VRSBENCH, UNKNOWN"
    )
    dataset_role: DatasetRole = Field(
        default=DatasetRole.INFERENCE,
        description="Operational role: INFERENCE, TRAINING, VALIDATION, BENCHMARK_EVALUATION, UNASSIGNED"
    )

    # Spatial
    width: int
    height: int
    crs: Optional[str] = Field(None, description="CRS as EPSG string or WKT")
    crs_epsg: Optional[int] = None
    transform: Optional[list[float]] = Field(None, description="Affine transform as 6 floats")
    bounds: Optional[dict | SpatialBounds] = Field(None, description="left, bottom, right, top")
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
    detection_method: str = Field(
        default="metadata_assisted_heuristic",
        description="Mechanism used to detect modality (metadata tag vs spectral heuristic)"
    )

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
