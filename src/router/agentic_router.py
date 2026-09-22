"""
SatQuery AI -- Agentic Dynamic Router
Section 9: Sensor-Aware Routing (SAR gets its own pathway)
Section 15: Agentic Router (Input validation, capability matching, fallback selection, refusal logic)
"""

from __future__ import annotations

import logging
from typing import Optional

from src.contracts.query_contracts import (
    PathwayType,
    QueryIntent,
    RoutingDecision,
    TaskType,
    ToolExecutionRecord,
    ToolExecutionStatus,
)
from src.contracts.raster_contracts import RasterMetadata, SensorModality
from src.router.capability_registry import CapabilityRegistry, RuntimeMode
from src.router.query_parser import QueryParser

logger = logging.getLogger("satquery.router")


class AgenticRouter:
    """
    Intelligent router that evaluates natural language intent alongside
    provided sensor data, dynamically directing tasks to truthful pathways
    or issuing precise refusals.
    """

    def __init__(
        self,
        parser: Optional[QueryParser] = None,
        registry: Optional[CapabilityRegistry] = None,
    ):
        self.parser = parser or QueryParser()
        self.registry = registry or CapabilityRegistry()

    @staticmethod
    def _build_tool_executions(tool_sequence: list[str]) -> list[ToolExecutionRecord]:
        """
        Distinguish SCHEDULED tools from those actually EXECUTED vs NOT_IMPLEMENTED.
        Enforces the Golden Rule: never claim an analysis tool was executed before its day.
        """
        records = []
        for tool in tool_sequence:
            base_name = tool.split(" ")[0].strip()
            if base_name in ("RasterInspector", "CompatibilityChecker"):
                records.append(ToolExecutionRecord(
                    tool_name=tool,
                    status=ToolExecutionStatus.EXECUTED,
                    details="Executed during raster ingestion & compatibility verification"
                ))
            else:
                records.append(ToolExecutionRecord(
                    tool_name=tool,
                    status=ToolExecutionStatus.NOT_IMPLEMENTED,
                    details="Workflow scheduled; execution engine scheduled for subsequent milestone"
                ))
        return records

    def route(
        self,
        query: str,
        input_files: list[RasterMetadata],
    ) -> RoutingDecision:
        """
        Produce a deterministic, transparent routing decision.
        """
        intent = self.parser.parse(query)
        provided_count = len(input_files)
        file_ids = [f.file_id for f in input_files]
        modalities = [f.modality.value for f in input_files]
        input_sources = [
            f.input_source.value if hasattr(f.input_source, "value") else str(f.input_source)
            for f in input_files
        ]
        dataset_roles = [
            f.dataset_role.value if hasattr(f.dataset_role, "value") else str(f.dataset_role)
            for f in input_files
        ]

        # -------------------------------------------------------------------
        # Rule 0: Zero input rasters
        # -------------------------------------------------------------------
        if provided_count == 0:
            return RoutingDecision(
                query=query,
                intent=intent,
                task_type=TaskType.UNSUPPORTED,
                selected_capability="none",
                pathway=PathwayType.REFUSAL,
                pathway_label="Refusal: Missing Input Imagery",
                is_executable=False,
                refusal_reason=(
                    "Cannot perform remote sensing analysis. "
                    "Reason: No satellite imagery was supplied. "
                    "Required: Upload at least one GeoTIFF or TIFF raster."
                ),
                required_inputs_count=1,
                provided_inputs_count=0,
                input_file_ids=[],
                input_modalities=[],
                input_sources=[],
                dataset_roles=[],
                tool_sequence=[],
                tool_executions=[],
                runtime_mode=self.registry.runtime_mode.value,
            )

        # -------------------------------------------------------------------
        # Rule 1: Temporal Change Request
        # -------------------------------------------------------------------
        if intent.requires_temporal_pair or intent.task_type == TaskType.TEMPORAL_CHANGE:
            if provided_count < 2:
                # Mandatory refusal per Section 15 & DEMO 6: Never hallucinate missing observation
                return RoutingDecision(
                    query=query,
                    intent=intent,
                    task_type=TaskType.TEMPORAL_CHANGE,
                    selected_capability="temporal_change",
                    pathway=PathwayType.REFUSAL,
                    pathway_label="Refusal: Insufficient Temporal Observations",
                    is_executable=False,
                    refusal_reason=(
                        f"Cannot perform temporal change analysis. "
                        f"Reason: Only one acquisition was supplied ({input_files[0].filename}). "
                        f"Required: Two temporally distinct observations of the same geographic area."
                    ),
                    required_inputs_count=2,
                    provided_inputs_count=provided_count,
                    input_file_ids=file_ids,
                    input_modalities=modalities,
                    input_sources=input_sources,
                    dataset_roles=dataset_roles,
                    tool_sequence=[],
                    tool_executions=[],
                    runtime_mode=self.registry.runtime_mode.value,
                )

            # Two observations supplied
            tools = [
                "CompatibilityChecker",
                "RegistrationQualityGate (AROSICS)",
                "TemporalConfoundCheck",
                "DeterministicChangeEngine",
                "L1L2DeclarationGate",
                "EvidenceStore",
            ]
            return RoutingDecision(
                query=query,
                intent=intent,
                task_type=TaskType.TEMPORAL_CHANGE,
                selected_capability="temporal_change",
                pathway=PathwayType.TEMPORAL_CHANGE_ENGINE,
                pathway_label="Bi-Temporal Change Engine [Registration Gate + L1/L2 Declaration]",
                is_executable=True,
                refusal_reason=None,
                required_inputs_count=2,
                provided_inputs_count=provided_count,
                input_file_ids=file_ids,
                input_modalities=modalities,
                input_sources=input_sources,
                dataset_roles=dataset_roles,
                tool_sequence=tools,
                tool_executions=self._build_tool_executions(tools),
                runtime_mode=self.registry.runtime_mode.value,
            )

        # -------------------------------------------------------------------
        # Rule 2: Optical-SAR Multimodal Fusion Request
        # -------------------------------------------------------------------
        if intent.requires_optical_sar or intent.task_type == TaskType.OPTICAL_SAR_ANALYSIS:
            has_optical = any(f.modality in (SensorModality.OPTICAL, SensorModality.MULTISPECTRAL) for f in input_files)
            has_sar = any(f.modality == SensorModality.SAR for f in input_files)

            if provided_count < 2 or not (has_optical and has_sar):
                return RoutingDecision(
                    query=query,
                    intent=intent,
                    task_type=TaskType.OPTICAL_SAR_ANALYSIS,
                    selected_capability="optical_sar_fusion",
                    pathway=PathwayType.REFUSAL,
                    pathway_label="Refusal: Missing Modality for Fusion",
                    is_executable=False,
                    refusal_reason=(
                        "Cannot perform Optical-SAR cross-modal fusion. "
                        f"Reason: Expected 1 Optical/Multispectral and 1 SAR raster. "
                        f"Supplied: {modalities}."
                    ),
                    required_inputs_count=2,
                    provided_inputs_count=provided_count,
                    input_file_ids=file_ids,
                    input_modalities=modalities,
                    input_sources=input_sources,
                    dataset_roles=dataset_roles,
                    tool_sequence=[],
                    tool_executions=[],
                    runtime_mode=self.registry.runtime_mode.value,
                )

            tools = [
                "CompatibilityChecker",
                "CommonGridResampler",
                "OpticalEvidenceExtractor (NDWI)",
                "SARBackscatterExtractor (Thresholding)",
                "CrossModalFusionEngine",
                "AgreementTierClassifier",
            ]
            return RoutingDecision(
                query=query,
                intent=intent,
                task_type=TaskType.OPTICAL_SAR_ANALYSIS,
                selected_capability="optical_sar_fusion",
                pathway=PathwayType.OPTICAL_SAR_FUSION,
                pathway_label="Cross-Modal Optical-SAR Fusion Engine [Agreement Tiers & Dual Evidence]",
                is_executable=True,
                refusal_reason=None,
                required_inputs_count=2,
                provided_inputs_count=provided_count,
                input_file_ids=file_ids,
                input_modalities=modalities,
                input_sources=input_sources,
                dataset_roles=dataset_roles,
                tool_sequence=tools,
                tool_executions=self._build_tool_executions(tools),
                runtime_mode=self.registry.runtime_mode.value,
            )

        # -------------------------------------------------------------------
        # Rule 3: Single Image Input -- Sensor Aware Routing
        # Section 9: SAR gets its own pathway! Never route SAR to optical VLM.
        # -------------------------------------------------------------------
        primary_file = input_files[0]

        if primary_file.modality == SensorModality.SAR:
            # Section 9: SAR strictly routed to SAR deterministic tools
            tools = [
                "RasterInspector",
                "SARBackscatterAnalysis (VV/VH stats)",
                "LeeSpeckleFilter",
                "PolarizationRatioEstimator",
                "SARStructuredResponseComposer",
            ]
            return RoutingDecision(
                query=query,
                intent=intent,
                task_type=TaskType.SINGLE_IMAGE_VQA_SAR,
                selected_capability="single_image_vqa_sar",
                pathway=PathwayType.SAR_DETERMINISTIC_TOOLS,
                pathway_label="SAR Deterministic Feature Pathway [Radar Backscatter Analysis -- No Optical Hallucination]",
                is_executable=True,
                refusal_reason=None,
                required_inputs_count=1,
                provided_inputs_count=1,
                input_file_ids=file_ids[:1],
                input_modalities=[primary_file.modality.value],
                input_sources=input_sources[:1],
                dataset_roles=dataset_roles[:1],
                tool_sequence=tools,
                tool_executions=self._build_tool_executions(tools),
                runtime_mode=self.registry.runtime_mode.value,
            )

        # Optical / Multispectral single image
        if intent.task_type == TaskType.SINGLE_IMAGE_GROUNDING:
            tools = [
                "RasterInspector",
                "TargetEntityDetector",
                "SpectralRegionOfInterestExtractor",
                "BoundingBoxCalculator",
                "EvidenceOverlayRenderer",
            ]
            return RoutingDecision(
                query=query,
                intent=intent,
                task_type=TaskType.SINGLE_IMAGE_GROUNDING,
                selected_capability="single_image_grounding",
                pathway=PathwayType.OPTICAL_DETERMINISTIC if self.registry.runtime_mode != RuntimeMode.FULL_AI else PathwayType.OPTICAL_VLM,
                pathway_label="Text-Guided Grounding [Geospatial ROI Localizer & Bounding Box]",
                is_executable=True,
                refusal_reason=None,
                required_inputs_count=1,
                provided_inputs_count=1,
                input_file_ids=file_ids[:1],
                input_modalities=[primary_file.modality.value],
                input_sources=input_sources[:1],
                dataset_roles=dataset_roles[:1],
                tool_sequence=tools,
                tool_executions=self._build_tool_executions(tools),
                runtime_mode=self.registry.runtime_mode.value,
            )

        # Optical VQA / Land cover / Caption
        pathway = (
            PathwayType.OPTICAL_VLM
            if self.registry.runtime_mode == RuntimeMode.FULL_AI
            else PathwayType.OPTICAL_DETERMINISTIC
        )
        pathway_label = (
            "Optical VLM Pathway [GeoChat-7B]"
            if pathway == PathwayType.OPTICAL_VLM
            else "Optical Deterministic Feature Pathway [Spectral Index Baseline]"
        )

        tools = [
            "RasterInspector",
            "SpectralIndexEngine (NDVI/NDWI)",
            "LandCoverClassProbabilityEstimator",
            "ResponseComposer",
        ]
        return RoutingDecision(
            query=query,
            intent=intent,
            task_type=TaskType.SINGLE_IMAGE_VQA_OPTICAL,
            selected_capability="single_image_vqa_optical",
            pathway=pathway,
            pathway_label=pathway_label,
            is_executable=True,
            refusal_reason=None,
            required_inputs_count=1,
            provided_inputs_count=1,
            input_file_ids=file_ids[:1],
            input_modalities=[primary_file.modality.value],
            input_sources=input_sources[:1],
            dataset_roles=dataset_roles[:1],
            tool_sequence=tools,
            tool_executions=self._build_tool_executions(tools),
            runtime_mode=self.registry.runtime_mode.value,
        )

