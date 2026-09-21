"""
SatQuery AI -- Query Contracts & Routing Schemas
Section 14: Query Contract & Task Categorization
Section 15: Agentic Router Schemas
"""

from __future__ import annotations

import enum
from typing import Optional
from pydantic import BaseModel, Field


class TaskType(str, enum.Enum):
    """Supported task types per Section 14."""
    SINGLE_IMAGE_VQA_OPTICAL = "single_image_vqa_optical"
    SINGLE_IMAGE_VQA_SAR = "single_image_vqa_sar"
    SINGLE_IMAGE_GROUNDING = "single_image_grounding"
    SINGLE_IMAGE_CAPTION = "single_image_caption"
    TEMPORAL_CHANGE = "temporal_change"
    OPTICAL_SAR_ANALYSIS = "optical_sar_analysis"
    UNSUPPORTED = "unsupported"


class QueryIntent(BaseModel):
    """
    Structured query intent extracted from natural language.
    Section 14: structured contract representing user intent.
    Note on confidence: Confidence values from the rule-based query parser
    are uncalibrated heuristic ranks, never calibrated Bayesian probabilities.
    """
    raw_query: str
    task_type: TaskType
    requires_temporal_pair: bool = False
    requires_optical_sar: bool = False
    requires_spatial_evidence: bool = False
    requested_outputs: list[str] = Field(default_factory=list)
    measurement_required: bool = False
    target_entities: list[str] = Field(default_factory=list)
    confidence_type: str = Field(
        default="heuristic_uncalibrated",
        description="Type of confidence metric (heuristic_uncalibrated vs calibrated)"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Heuristic match score; NOT a calibrated probability."
    )


class PathwayType(str, enum.Enum):
    """Truthfully declared execution pathway per Section 0 & Section 9."""
    OPTICAL_VLM = "optical_vlm"
    OPTICAL_DETERMINISTIC = "optical_deterministic"
    SAR_DETERMINISTIC_TOOLS = "sar_deterministic_tools"
    SAR_INPUT_ADAPTER_VLM = "sar_input_adapter_vlm"
    TEMPORAL_CHANGE_ENGINE = "temporal_change_engine"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"
    REFUSAL = "refusal"


class ToolExecutionStatus(str, enum.Enum):
    """Execution state of an individual tool in the pipeline."""
    SCHEDULED = "SCHEDULED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    BLOCKED = "BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"


class ToolExecutionRecord(BaseModel):
    """Record of an individual tool scheduled or executed in a pipeline."""
    tool_name: str
    status: ToolExecutionStatus = ToolExecutionStatus.SCHEDULED
    details: Optional[str] = None


class RoutingDecision(BaseModel):
    """
    Dynamic routing decision emitted by the Agentic Router.
    Section 15: Task, inputs, capability, availability, fallback, tool sequence.
    """
    query: str
    intent: QueryIntent
    task_type: TaskType
    selected_capability: str
    pathway: PathwayType
    pathway_label: str = Field(..., description="Human-readable transparent pathway description")
    is_executable: bool
    refusal_reason: Optional[str] = None
    required_inputs_count: int = 1
    provided_inputs_count: int = 0
    input_file_ids: list[str] = Field(default_factory=list)
    input_modalities: list[str] = Field(default_factory=list)
    tool_sequence: list[str] = Field(default_factory=list, description="Tool names scheduled")
    tool_executions: list[ToolExecutionRecord] = Field(
        default_factory=list,
        description="Separated execution states for each tool (SCHEDULED, EXECUTED, NOT_IMPLEMENTED)"
    )
    runtime_mode: str = "DEMO_FALLBACK"
    execution_trace_id: Optional[str] = None


class AnalysisResult(BaseModel):
    """
    Result contract declaring the truthful mechanism that produced the answer.
    Enforces the Honesty Rule: declared mechanism, model name, confidence, and limitations.
    """
    task: TaskType
    mechanism: Optional[str] = Field(
        None,
        description="e.g. deterministic_sar_analysis, deterministic_optical_spectral_analysis, geochat_vlm"
    )
    model: Optional[str] = None
    answer: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)
    confidence: dict = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    status: str = "ROUTED_PENDING_EXECUTION"
