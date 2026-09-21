"""
SatQuery AI -- Capability Registry
Section 10 (Runtime Modes) & Section 23 (Capability State Machine)
"""

from __future__ import annotations

import enum
from typing import Optional
from pydantic import BaseModel, Field


class RuntimeMode(str, enum.Enum):
    """Runtime execution mode determined by system compute profile."""
    FULL_AI = "FULL_AI"
    HYBRID = "HYBRID"
    DEMO_FALLBACK = "DEMO_FALLBACK"


class CapabilityStatus(str, enum.Enum):
    """Status of an individual engine capability."""
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"
    BLOCKED_LICENSE = "BLOCKED_LICENSE"
    DISABLED = "DISABLED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    FALLBACK_ACTIVE = "FALLBACK_ACTIVE"


class CapabilityRecord(BaseModel):
    name: str
    description: str
    status: CapabilityStatus
    primary_engine: str
    fallback_engine: Optional[str] = None
    license_note: Optional[str] = None
    requires_gpu: bool = False


class CapabilityRegistry:
    """
    Central registry tracking operational status of all SatQuery AI capabilities.
    Maintains runtime mode and transparent fallback mapping per the Honesty Rule.
    """

    def __init__(self, runtime_mode: RuntimeMode = RuntimeMode.DEMO_FALLBACK):
        self.runtime_mode = runtime_mode
        self._capabilities: dict[str, CapabilityRecord] = {}
        self._init_defaults()

    def _init_defaults(self):
        # Core data gateways
        self.register(CapabilityRecord(
            name="upload",
            description="Raster file upload, format integrity, and metadata inspection",
            status=CapabilityStatus.READY,
            primary_engine="RasterInspector (Rasterio/GDAL)",
            requires_gpu=False,
        ))
        self.register(CapabilityRecord(
            name="metadata_inspection",
            description="GeoTIFF metadata extraction (CRS, transform, bands, bounds)",
            status=CapabilityStatus.READY,
            primary_engine="RasterInspector",
            requires_gpu=False,
        ))
        self.register(CapabilityRecord(
            name="compatibility_check",
            description="Data readiness and pair spatial/spectral compatibility checking",
            status=CapabilityStatus.READY,
            primary_engine="CompatibilityChecker",
            requires_gpu=False,
        ))

        # Optical VQA
        self.register(CapabilityRecord(
            name="single_image_vqa_optical",
            description="Optical remote sensing visual question answering",
            status=CapabilityStatus.FALLBACK_ACTIVE if self.runtime_mode != RuntimeMode.FULL_AI else CapabilityStatus.READY,
            primary_engine="GeoChat (7B RS-VLM)",
            fallback_engine="Deterministic Optical Feature Extractor",
            license_note="GeoChat Apache-2.0 adapter over Llama-2 base (attribution required)",
            requires_gpu=True,
        ))

        # SAR VQA -- Always independent of GPU / Optical VLM
        self.register(CapabilityRecord(
            name="single_image_vqa_sar",
            description="Single-image SAR analysis and radar backscatter VQA",
            status=CapabilityStatus.READY,
            primary_engine="SAR Deterministic Feature Tools (Section 8.5)",
            fallback_engine=None,
            license_note="Native internal deterministic engine; zero optical hallucination",
            requires_gpu=False,
        ))

        # Grounding
        self.register(CapabilityRecord(
            name="single_image_grounding",
            description="Text-guided geospatial object detection and bounding box localization",
            status=CapabilityStatus.FALLBACK_ACTIVE if self.runtime_mode != RuntimeMode.FULL_AI else CapabilityStatus.READY,
            primary_engine="GeoChat Grounding Head",
            fallback_engine="Spectral Region-of-Interest (ROI) Extractor",
            requires_gpu=True,
        ))

        # Temporal Change Detection
        self.register(CapabilityRecord(
            name="temporal_change",
            description="Bi-temporal change detection with registration gate and L1/L2 distinction",
            status=CapabilityStatus.READY,
            primary_engine="Deterministic Change Engine + AROSICS Gate",
            fallback_engine=None,
            requires_gpu=False,
        ))

        # Optical-SAR Multimodal Fusion
        self.register(CapabilityRecord(
            name="optical_sar_fusion",
            description="Cross-modal optical and SAR joint surface water and land analysis",
            status=CapabilityStatus.READY,
            primary_engine="Co-registered Spatial Fusion Engine (Section 18)",
            fallback_engine=None,
            requires_gpu=False,
        ))

        # Specialist models explicitly tracked
        self.register(CapabilityRecord(
            name="geochat",
            description="GeoChat 7B multimodal RS foundation model",
            status=CapabilityStatus.UNAVAILABLE if self.runtime_mode == RuntimeMode.DEMO_FALLBACK else CapabilityStatus.READY,
            primary_engine="MBZUAI GeoChat-7B",
            license_note="Apache-2.0 / Llama-2 License",
            requires_gpu=True,
        ))
        self.register(CapabilityRecord(
            name="changechat",
            description="ChangeChat bi-temporal RS-VLM",
            status=CapabilityStatus.BLOCKED_LICENSE,
            primary_engine="ChangeChat Model",
            license_note="License terms require audit clearance",
            requires_gpu=True,
        ))
        self.register(CapabilityRecord(
            name="croma",
            description="CROMA cross-modal optical-SAR self-supervised representations",
            status=CapabilityStatus.DISABLED,
            primary_engine="CROMA Pretrained Checkpoint",
            requires_gpu=True,
        ))

    def register(self, record: CapabilityRecord):
        self._capabilities[record.name] = record

    def get(self, name: str) -> Optional[CapabilityRecord]:
        return self._capabilities.get(name)

    def get_all(self) -> dict[str, dict]:
        return {
            name: {
                "name": cap.name,
                "description": cap.description,
                "status": cap.status.value,
                "primary_engine": cap.primary_engine,
                "fallback_engine": cap.fallback_engine,
                "requires_gpu": cap.requires_gpu,
                "license_note": cap.license_note,
            }
            for name, cap in self._capabilities.items()
        }

    def is_ready(self, name: str) -> bool:
        cap = self.get(name)
        if not cap:
            return False
        return cap.status in (CapabilityStatus.READY, CapabilityStatus.FALLBACK_ACTIVE)
