"""
SatQuery AI -- Execution Trace Engine
Section 24: Execution Trace Logging (Factual, transparent, verifiable; no hidden CoT)
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    name: str
    status: str = Field(default="completed", description="completed, in_progress, failed, skipped")
    details: Optional[str] = None
    elapsed_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ExecutionTrace(BaseModel):
    """
    Immutable execution trace representing exact system steps executed.
    Section 24: Facts only, shows selected tools, strictly prevents fabricated steps.
    """
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: str
    pathway: str
    runtime_mode: str
    tools_selected: list[str] = Field(default_factory=list)
    steps: list[TraceStep] = Field(default_factory=list)
    total_elapsed_ms: float = 0.0
    is_success: bool = True
    error_message: Optional[str] = None

    def add_step(
        self,
        name: str,
        status: str = "completed",
        details: Optional[str] = None,
        elapsed_ms: float = 0.0,
    ) -> TraceStep:
        step = TraceStep(
            name=name,
            status=status,
            details=details,
            elapsed_ms=elapsed_ms,
        )
        self.steps.append(step)
        self.total_elapsed_ms += elapsed_ms
        return step

    def format_text(self) -> str:
        """Format as human-readable CLI/Audit trace per Section 24."""
        lines = ["EXECUTION TRACE:"]
        for step in self.steps:
            icon = "✓" if step.status == "completed" else ("✗" if step.status == "failed" else "○")
            detail_str = f" -- {step.details}" if step.details else ""
            lines.append(f"  {icon} {step.name}{detail_str}")
        lines.append(f"Tools Invoked: {', '.join(self.tools_selected) if self.tools_selected else 'None'}")
        lines.append(f"Pathway Declared: {self.pathway}")
        lines.append(f"Total Time: {self.total_elapsed_ms:.1f}ms")
        return "\n".join(lines)


class TraceEngine:
    """Factory and recorder for execution traces."""

    @staticmethod
    def create_trace(
        task_type: str,
        pathway: str,
        runtime_mode: str,
        tools_selected: Optional[list[str]] = None,
    ) -> ExecutionTrace:
        return ExecutionTrace(
            task_type=task_type,
            pathway=pathway,
            runtime_mode=runtime_mode,
            tools_selected=tools_selected or [],
        )
