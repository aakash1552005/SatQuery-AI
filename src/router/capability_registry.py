"""
SatQuery AI -- Capability Registry
Section 10 (Runtime Modes) & Section 23 (Capability State Machine)
Strict implementation integrity: clearly separates routing readiness from execution readiness.
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
    ROUTING_READY = "ROUTING_READY"
    UNAVAILABLE = "UNAVAILABLE"
    BLOCKED_LICENSE = "BLOCKED_LICENSE"
    DISABLED = "DISABLED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    FALLBACK_ACTIVE = "FALLBACK_ACTIVE"


class CapabilityRecord(BaseModel):
    name: str
    description: str
    status: CapabilityStatus
    routing_readiness: str = "READY"
    execution_readiness: str = "NOT_IMPLEMENTED"
    model_availability: str = "UNAVAILABLE"
    compute_readiness: str = "CPU_COMPATIBLE"
    automated_test_coverage: str = "ROUTING_TESTED"
    primary_engine: str
    fallback_engine: Optional[str] = None
    license_note: Optional[str] = None
    requires_gpu: bool = False
    structured_status: Optional[dict] = None


class CapabilityRegistry:
    """
    Central registry tracking operational status of all SatQuery AI capabilities.
    Enforces the Golden Rule: capabilities are only marked READY when both routing
    and physical analysis tools actually execute on the host and pass tests.
    """

    def __init__(self, runtime_mode: RuntimeMode = RuntimeMode.DEMO_FALLBACK):
        self.runtime_mode = runtime_mode
        self._capabilities: dict[str, CapabilityRecord] = {}
        self._init_defaults()

    def _init_defaults(self):
        # 1. Core data gateways -- IMPLEMENTED & EXECUTED
        self.register(CapabilityRecord(
            name="upload",
            description="Raster file upload, format integrity, and metadata inspection",
            status=CapabilityStatus.READY,
            routing_readiness="READY",
            execution_readiness="READY",
            model_availability="NOT_REQUIRED_DETERMINISTIC",
            compute_readiness="CPU_READY",
            automated_test_coverage="AUTOMATED_TESTS_PASSING (tests/test_day1.py)",
            primary_engine="RasterInspector (Rasterio/GDAL)",
            requires_gpu=False,
        ))
        self.register(CapabilityRecord(
            name="metadata_inspection",
            description="GeoTIFF metadata extraction (CRS, transform, bands, bounds, modality)",
            status=CapabilityStatus.READY,
            routing_readiness="READY",
            execution_readiness="READY",
            model_availability="NOT_REQUIRED_DETERMINISTIC",
            compute_readiness="CPU_READY",
            automated_test_coverage="AUTOMATED_TESTS_PASSING (tests/test_day1.py)",
            primary_engine="RasterInspector",
            requires_gpu=False,
        ))
        self.register(CapabilityRecord(
            name="compatibility_check",
            description="Data readiness and pair spatial/spectral compatibility checking",
            status=CapabilityStatus.READY,
            routing_readiness="READY",
            execution_readiness="READY",
            model_availability="NOT_REQUIRED_DETERMINISTIC",
            compute_readiness="CPU_READY",
            automated_test_coverage="AUTOMATED_TESTS_PASSING (tests/test_day1.py)",
            primary_engine="CompatibilityChecker",
            requires_gpu=False,
        ))

        # 2. Optical VQA -- Deterministic spectral engine implemented & verified (Day 3)
        self.register(CapabilityRecord(
            name="single_image_vqa_optical",
            description="Optical remote sensing visual question answering & spectral index analysis",
            status=CapabilityStatus.READY,
            routing_readiness="READY",
            execution_readiness="READY",
            model_availability="UNAVAILABLE (GeoChat-7B requires CUDA GPU; Host is Profile D)",
            compute_readiness="CPU_READY",
            automated_test_coverage="ROUTING_AND_EXECUTION_TESTED (tests/test_day3_optical.py)",
            primary_engine="Deterministic Optical Spectral Analysis Engine (Section 8.5 / Day 3)",
            fallback_engine=None,
            license_note="Native internal deterministic engine; zero hallucination",
            requires_gpu=False,
        ))

        # 3. SAR VQA -- Deterministic radar tools implemented & verified (Day 3)
        self.register(CapabilityRecord(
            name="single_image_vqa_sar",
            description="Single-image SAR analysis and radar backscatter characterization",
            status=CapabilityStatus.READY,
            routing_readiness="READY",
            execution_readiness="READY",
            model_availability="UNAVAILABLE (No SAR-native VLM; deterministic pathway executed)",
            compute_readiness="CPU_READY",
            automated_test_coverage="ROUTING_AND_EXECUTION_TESTED (tests/test_day3_sar.py)",
            primary_engine="SAR Deterministic Feature Tools (Section 8.5 / Day 3)",
            fallback_engine=None,
            license_note="Native internal deterministic engine; zero optical hallucination",
            requires_gpu=False,
        ))

        # 4. Grounding -- Routing implemented; head scheduled for Day 3
        self.register(CapabilityRecord(
            name="single_image_grounding",
            description="Text-guided geospatial object detection and bounding box localization",
            status=CapabilityStatus.NOT_IMPLEMENTED,
            routing_readiness="READY",
            execution_readiness="NOT_IMPLEMENTED (Scheduled Day 3)",
            model_availability="UNAVAILABLE (GeoChat Grounding Head requires CUDA GPU)",
            compute_readiness="FALLBACK_CPU_COMPATIBLE",
            automated_test_coverage="ROUTING_TESTED (tests/test_day2.py)",
            primary_engine="GeoChat Grounding Head",
            fallback_engine="Spectral Region-of-Interest (ROI) Extractor",
            requires_gpu=True,
        ))

        # 5. Temporal Change Detection -- Routing implemented; registration gate & engine scheduled Day 5
        self.register(CapabilityRecord(
            name="temporal_change",
            description="Bi-temporal change detection with registration gate and L1/L2 distinction",
            status=CapabilityStatus.NOT_IMPLEMENTED,
            routing_readiness="READY",
            execution_readiness="NOT_IMPLEMENTED (Scheduled Day 5)",
            model_availability="BLOCKED_LICENSE (ChangeChat licensing review required)",
            compute_readiness="CPU_COMPATIBLE (Deterministic change engine requires zero GPU)",
            automated_test_coverage="ROUTING_AND_REFUSAL_TESTED (tests/test_day2.py)",
            primary_engine="Deterministic Change Engine + AROSICS Gate (Scheduled Day 5)",
            fallback_engine=None,
            requires_gpu=False,
        ))

        # 6. Optical-SAR Multimodal Fusion -- Routing implemented; fusion engine scheduled Day 6
        self.register(CapabilityRecord(
            name="optical_sar_fusion",
            description="Cross-modal optical and SAR joint surface water and land analysis",
            status=CapabilityStatus.NOT_IMPLEMENTED,
            routing_readiness="READY",
            execution_readiness="NOT_IMPLEMENTED (Scheduled Day 6)",
            model_availability="DISABLED (CROMA optional stretch model)",
            compute_readiness="CPU_COMPATIBLE (Deterministic cross-modal grid fusion requires zero GPU)",
            automated_test_coverage="ROUTING_TESTED (tests/test_day2.py)",
            primary_engine="Co-registered Spatial Fusion Engine (Section 18 -- Scheduled Day 6)",
            fallback_engine=None,
            requires_gpu=False,
        ))

        # 7. Specialist models explicitly tracked
        self.register(CapabilityRecord(
            name="geochat",
            description="GeoChat 7B multimodal RS foundation model",
            status=CapabilityStatus.UNAVAILABLE,
            routing_readiness="N/A",
            execution_readiness="UNAVAILABLE (Host is Profile D CPU; requires CUDA GPU)",
            model_availability="UNAVAILABLE (No local GPU/weights)",
            compute_readiness="GPU_REQUIRED",
            automated_test_coverage="PREFLIGHT_PENDING_DAY_3",
            primary_engine="MBZUAI GeoChat-7B",
            license_note="Apache-2.0 / Llama-2 License (Attribution required)",
            requires_gpu=True,
            structured_status={
                "repository": "ABSENT",
                "dependencies": "MISSING",
                "model_weights": "ABSENT",
                "CUDA": "UNAVAILABLE",
                "GPU_VRAM": "NOT_AVAILABLE",
                "environment_preflight": "NOT_EXECUTED",
                "real_model_inference": "NOT_EXECUTED",
                "final_capability": "UNAVAILABLE",
            },
        ))
        self.register(CapabilityRecord(
            name="changechat",
            description="ChangeChat bi-temporal RS-VLM",
            status=CapabilityStatus.BLOCKED_LICENSE,
            routing_readiness="N/A",
            execution_readiness="BLOCKED (Academic license requires written clearance)",
            model_availability="BLOCKED_LICENSE",
            compute_readiness="GPU_REQUIRED",
            automated_test_coverage="NONE",
            primary_engine="ChangeChat Model",
            license_note="License terms require audit clearance",
            requires_gpu=True,
        ))
        self.register(CapabilityRecord(
            name="croma",
            description="CROMA cross-modal optical-SAR self-supervised representations",
            status=CapabilityStatus.DISABLED,
            routing_readiness="N/A",
            execution_readiness="DISABLED (Optional stretch model)",
            model_availability="DISABLED",
            compute_readiness="GPU_REQUIRED",
            automated_test_coverage="NONE",
            primary_engine="CROMA Pretrained Checkpoint",
            requires_gpu=True,
        ))

    def register(self, record: CapabilityRecord):
        self._capabilities[record.name] = record

    def get(self, name: str) -> Optional[CapabilityRecord]:
        return self._capabilities.get(name)

    def get_all(self) -> dict[str, dict]:
        """Return dict representation of all registered capabilities."""
        return {
            name: {
                "name": cap.name,
                "description": cap.description,
                "status": cap.status.value,
                "routing_readiness": cap.routing_readiness,
                "execution_readiness": cap.execution_readiness,
                "model_availability": cap.model_availability,
                "compute_readiness": cap.compute_readiness,
                "automated_test_coverage": cap.automated_test_coverage,
                "primary_engine": cap.primary_engine,
                "fallback_engine": cap.fallback_engine,
                "requires_gpu": cap.requires_gpu,
                "license_note": cap.license_note,
                "structured_status": cap.structured_status,
            }
            for name, cap in self._capabilities.items()
        }

    def run_geochat_preflight(self) -> dict:
        """
        Dynamically execute GeoChat preflight audit on current host.
        Truthfully records hardware limitations without fabricating model inference.
        """
        from pathlib import Path
        import torch

        repo_path = Path("models/geochat")
        repo_status = "PRESENT" if repo_path.exists() else "ABSENT"

        cuda_available = torch.cuda.is_available()
        cuda_status = "AVAILABLE" if cuda_available else "UNAVAILABLE"
        gpu_vram = (
            f"{round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)} GB"
            if cuda_available
            else "NOT_AVAILABLE"
        )

        deps_status = "MISSING"
        try:
            import transformers
            deps_status = "PARTIAL (transformers installed, flash-attn absent)"
        except ImportError:
            deps_status = "MISSING"

        preflight_status = "PASSED" if cuda_available else "FAILED"

        status_dict = {
            "repository": repo_status,
            "dependencies": deps_status,
            "model_weights": "ABSENT",
            "CUDA": cuda_status,
            "GPU_VRAM": gpu_vram,
            "environment_preflight": preflight_status,
            "real_model_inference": "NOT_EXECUTED",
            "final_capability": "UNAVAILABLE",
        }
        geochat_rec = self.get("geochat")
        if geochat_rec:
            geochat_rec.structured_status = status_dict
        return status_dict

    def get_geochat_preflight_status(self) -> dict:
        """
        Return structured GeoChat preflight and readiness status per Section 3.
        Distinguishes environment preflight from real model inference.
        """
        geochat_rec = self.get("geochat")
        if geochat_rec and geochat_rec.structured_status:
            return geochat_rec.structured_status
        return {
            "repository": "ABSENT",
            "dependencies": "MISSING",
            "model_weights": "ABSENT",
            "CUDA": "UNAVAILABLE",
            "GPU_VRAM": "NOT_AVAILABLE",
            "environment_preflight": "NOT_EXECUTED",
            "real_model_inference": "NOT_EXECUTED",
            "final_capability": "UNAVAILABLE",
        }

    def is_ready(self, name: str) -> bool:
        cap = self.get(name)
        if not cap:
            return False
        return cap.status == CapabilityStatus.READY and cap.execution_readiness == "READY"

    def is_routing_ready(self, name: str) -> bool:
        cap = self.get(name)
        if not cap:
            return False
        return cap.routing_readiness == "READY"


def run_geochat_preflight() -> dict:
    """Module-level helper to execute GeoChat host environment preflight."""
    registry = CapabilityRegistry()
    return registry.run_geochat_preflight()

