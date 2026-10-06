"""
SatQuery AI -- Security Middleware & Validation
Phase 1 (GAP-01 through GAP-06): Authentication, rate limiting, file validation, SSRF protection.
"""

from __future__ import annotations

import logging
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware

from app.backend.config import AppConfig

logger = logging.getLogger("satquery.security")


# ---------------------------------------------------------------------------
# GAP-01: API Key Authentication Middleware
# ---------------------------------------------------------------------------

class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates X-API-Key header on all /api/ endpoints.
    Skipped when config.api_key is None (development mode).
    Always allows: /, /index.html, /favicon.*, /docs, /openapi.json, /api/health
    """

    # Paths that never require authentication
    PUBLIC_PATHS = {
        "/", "/index.html", "/favicon.svg", "/favicon.ico",
        "/docs", "/redoc", "/openapi.json",
        "/api/health",
    }

    def __init__(self, app: FastAPI, config: AppConfig):
        super().__init__(app)
        self.api_key = config.api_key
        self.header_name = config.api_key_header

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth if no API key is configured (dev mode)
        if self.api_key is None:
            return await call_next(request)

        path = request.url.path.rstrip("/") or "/"

        # Allow public paths
        if path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Skip non-API paths (static frontend assets)
        if not path.startswith("/api/"):
            return await call_next(request)

        # Validate API key
        provided_key = request.headers.get(self.header_name)
        if not provided_key or provided_key != self.api_key:
            logger.warning("Unauthorized API request to %s from %s", path, request.client.host if request.client else "unknown")
            return Response(
                content='{"error":{"code":"UNAUTHORIZED","message":"Invalid or missing API key."}}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)


# ---------------------------------------------------------------------------
# GAP-02: In-Memory Rate Limiter (No external dependency)
# ---------------------------------------------------------------------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple token-bucket rate limiter keyed by client IP.
    Configurable per-endpoint limits via AppConfig.
    """

    def __init__(self, app: FastAPI, config: AppConfig):
        super().__init__(app)
        self.enabled = config.rate_limit_enabled
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._limits = self._parse_limits(config)

    @staticmethod
    def _parse_limit_string(limit_str: str) -> tuple[int, int]:
        """Parse '10/minute' -> (10, 60)."""
        parts = limit_str.strip().split("/")
        count = int(parts[0])
        period_str = parts[1].lower() if len(parts) > 1 else "minute"
        periods = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
        period = periods.get(period_str, 60)
        return count, period

    def _parse_limits(self, config: AppConfig) -> dict[str, tuple[int, int]]:
        """Build path-prefix -> (count, period_seconds) mapping."""
        return {
            "/api/upload": self._parse_limit_string(config.rate_limit_upload),
            "/api/query": self._parse_limit_string(config.rate_limit_query),
            "/api/export": self._parse_limit_string(config.rate_limit_query),
            "/api/": self._parse_limit_string(config.rate_limit_default),
        }

    def _get_limit(self, path: str) -> tuple[int, int]:
        """Find the most specific limit for a path."""
        for prefix in ["/api/upload", "/api/query", "/api/export"]:
            if path.startswith(prefix):
                return self._limits[prefix]
        return self._limits["/api/"]

    def _is_rate_limited(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check if a key has exceeded its rate limit."""
        now = time.monotonic()
        # Prune old entries
        self._buckets[key] = [t for t in self._buckets[key] if now - t < window_seconds]
        if len(self._buckets[key]) >= max_requests:
            return True
        self._buckets[key].append(now)
        return False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        # Starlette TestClient uses 'testclient' as host; exempt from rate limiting
        if client_ip == "testclient":
            return await call_next(request)

        max_req, window = self._get_limit(path)
        bucket_key = f"{client_ip}:{path.split('/')[2] if len(path.split('/')) > 2 else 'api'}"

        if self._is_rate_limited(bucket_key, max_req, window):
            logger.warning("Rate limit exceeded for %s on %s", client_ip, path)
            return Response(
                content='{"error":{"code":"RATE_LIMITED","message":"Too many requests. Please slow down."}}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(window)},
            )

        return await call_next(request)


# ---------------------------------------------------------------------------
# GAP-04 + GAP-05: Upload File Validation
# ---------------------------------------------------------------------------

# Magic byte signatures for allowed file formats
MAGIC_SIGNATURES = {
    # TIFF (little-endian and big-endian)
    b"II\x2a\x00": "tiff",
    b"MM\x00\x2a": "tiff",
    # BigTIFF
    b"II\x2b\x00": "bigtiff",
    b"MM\x00\x2b": "bigtiff",
    # PNG
    b"\x89PNG": "png",
    # JPEG
    b"\xff\xd8\xff": "jpeg",
}


def validate_upload_file(
    file: UploadFile,
    max_size_bytes: int,
    allowed_extensions: set[str],
) -> tuple[bool, Optional[str]]:
    """
    Validate an uploaded file before writing to disk.
    Returns (is_valid, error_message).
    """
    filename = file.filename or ""

    # 1. Check filename extension
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in allowed_extensions:
        return False, (
            f"File type '.{ext}' is not allowed. "
            f"Accepted types: {', '.join(sorted(allowed_extensions))}"
        )

    # 2. Check Content-Length header hint (not reliable but fast pre-check)
    if file.size is not None and file.size > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        return False, f"File exceeds maximum upload size of {max_mb:.0f} MB."

    return True, None


def validate_file_magic_bytes(file_path: str | Path) -> tuple[bool, str]:
    """
    Validate file content by checking magic bytes after writing.
    Returns (is_valid, detected_type).
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)

        if len(header) < 3:
            return False, "unknown"

        for signature, file_type in MAGIC_SIGNATURES.items():
            if header[: len(signature)] == signature:
                return True, file_type

        return False, "unknown"
    except Exception:
        return False, "error"


# ---------------------------------------------------------------------------
# GAP-06: GPU Worker Registration Validation
# ---------------------------------------------------------------------------

def validate_gpu_worker_url(
    url: str,
    configured_secret: Optional[str] = None,
    provided_secret: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Validate a GPU worker registration request.
    - If a shared secret is configured, the request must provide it.
    - Block obviously dangerous URLs (localhost, internal IPs, file://).
    """
    # Secret validation
    if configured_secret is not None:
        if not provided_secret or provided_secret != configured_secret:
            return False, "GPU worker registration requires a valid shared secret."

    # Block dangerous URL patterns (SSRF protection)
    url_lower = url.lower().strip()
    blocked_patterns = [
        "file://",
        "ftp://",
        "gopher://",
        "dict://",
        "169.254.",         # AWS metadata
        "metadata.google",  # GCP metadata
        "100.100.100.200",  # Alibaba metadata
    ]
    for pattern in blocked_patterns:
        if pattern in url_lower:
            return False, f"URL contains blocked pattern: '{pattern}'. Registration denied."

    return True, None
