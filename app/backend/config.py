"""
SatQuery AI -- Centralized Application Configuration
Phase 2 (GAP-19): Environment-based configuration management.

All previously hardcoded values are now configurable via environment
variables or a .env file. Defaults are safe for local development.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Single source of truth for version (GAP-13)
# ---------------------------------------------------------------------------
__version__ = "1.0.0"


class AppConfig(BaseSettings):
    """
    Application-wide settings loaded from environment variables.
    Prefix: SATQUERY_
    """

    # Version (read-only, derived from __version__)
    version: str = Field(default=__version__, description="Application version")

    # Server
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, description="Server bind port")
    reload: bool = Field(default=False, description="Enable hot-reload (dev only)")
    log_level: str = Field(default="INFO", description="Logging level")

    # Security -- API Key Authentication (GAP-01)
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key for authenticating requests. "
            "If None, authentication is disabled (dev mode)."
        ),
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="HTTP header name for API key",
    )

    # Security -- CORS (GAP-01)
    cors_origins: str = Field(
        default="*",
        description=(
            "Comma-separated list of allowed CORS origins. "
            "Use '*' only for development. "
            "Example: 'https://satquery-ai.aakash1552005.workers.dev,http://localhost:3000'"
        ),
    )

    # Security -- Rate Limiting (GAP-02)
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting",
    )
    rate_limit_upload: str = Field(
        default="10/minute",
        description="Upload endpoint rate limit (format: 'count/period')",
    )
    rate_limit_query: str = Field(
        default="30/minute",
        description="Query endpoint rate limit",
    )
    rate_limit_default: str = Field(
        default="60/minute",
        description="Default rate limit for all other endpoints",
    )

    # Uploads (GAP-04, GAP-18)
    upload_dir: str = Field(
        default="data/uploads",
        description="Directory for uploaded files",
    )
    max_upload_size_mb: int = Field(
        default=50,
        description="Maximum file upload size in megabytes",
    )
    upload_ttl_hours: int = Field(
        default=24,
        description="Hours before uploaded files are automatically cleaned up (0 = never)",
    )

    # Allowed file types (GAP-05) -- MIME magic byte prefixes
    allowed_extensions: str = Field(
        default=".tif,.tiff,.geotiff,.png,.jpg,.jpeg",
        description="Comma-separated allowed file extensions for upload",
    )

    # Cloud GPU Bridge (GAP-06)
    gpu_worker_url: Optional[str] = Field(
        default=None,
        description="Pre-configured remote GPU worker URL",
    )
    gpu_worker_secret: Optional[str] = Field(
        default=None,
        description=(
            "Shared secret for GPU worker registration. "
            "If set, the /api/cloud-gpu/register endpoint requires this secret."
        ),
    )

    model_config = {
        "env_prefix": "SATQUERY_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    # ---- Derived helpers ----

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {ext.strip().lower().lstrip(".") for ext in self.allowed_extensions.split(",")}

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


# Singleton instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create the singleton AppConfig instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
