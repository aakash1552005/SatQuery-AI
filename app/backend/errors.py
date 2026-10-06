"""
SatQuery AI -- Standardized Error Responses
Phase 2 (GAP-08): Consistent error schema across all API endpoints.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Standardized error response body."""
    code: str = Field(..., description="Machine-readable error code (e.g., 'UPLOAD_TOO_LARGE')")
    message: str = Field(..., description="Human-readable error message")
    trace_id: str = Field(..., description="Unique identifier for this error occurrence")
    details: Optional[str] = Field(None, description="Additional context or debugging hints")


class ErrorResponse(BaseModel):
    """Top-level error wrapper."""
    error: ErrorDetail


def make_error_response(
    status_code: int,
    code: str,
    message: str,
    details: Optional[str] = None,
) -> JSONResponse:
    """Create a standardized JSON error response."""
    trace_id = str(uuid.uuid4())[:12]
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                code=code,
                message=message,
                trace_id=trace_id,
                details=details,
            )
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Pre-defined error codes
# ---------------------------------------------------------------------------

# Authentication
UNAUTHORIZED = ("UNAUTHORIZED", 401, "Invalid or missing API key.")
FORBIDDEN = ("FORBIDDEN", 403, "You do not have permission to perform this action.")

# Rate Limiting
RATE_LIMITED = ("RATE_LIMITED", 429, "Too many requests. Please slow down.")

# Upload Errors
UPLOAD_NO_FILENAME = ("UPLOAD_NO_FILENAME", 400, "No filename provided.")
UPLOAD_TOO_LARGE = ("UPLOAD_TOO_LARGE", 413, "File exceeds maximum upload size.")
UPLOAD_INVALID_TYPE = ("UPLOAD_INVALID_TYPE", 415, "File type is not supported.")
UPLOAD_SAVE_FAILED = ("UPLOAD_SAVE_FAILED", 500, "Failed to save uploaded file.")
UPLOAD_INVALID_CONTENT = ("UPLOAD_INVALID_CONTENT", 422, "File content does not match its extension.")

# Resource Errors
FILE_NOT_FOUND = ("FILE_NOT_FOUND", 404, "Referenced file not found.")
QUERY_EMPTY = ("QUERY_EMPTY", 400, "Query cannot be empty.")

# GPU Worker Errors
GPU_WORKER_INVALID_URL = ("GPU_WORKER_INVALID_URL", 400, "Invalid GPU worker URL.")
GPU_WORKER_UNREACHABLE = ("GPU_WORKER_UNREACHABLE", 502, "Could not reach remote GPU worker.")
GPU_WORKER_SECRET_REQUIRED = ("GPU_WORKER_SECRET_REQUIRED", 403, "GPU worker registration requires a valid secret.")


# ---------------------------------------------------------------------------
# Exception Handlers for FastAPI app
# ---------------------------------------------------------------------------

async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Wrap FastAPI HTTPException into standardized error format."""
    return make_error_response(
        status_code=exc.status_code,
        code=f"HTTP_{exc.status_code}",
        message=str(exc.detail),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions."""
    return make_error_response(
        status_code=500,
        code="INTERNAL_ERROR",
        message="An unexpected internal error occurred.",
        details=str(exc) if str(exc) else None,
    )
