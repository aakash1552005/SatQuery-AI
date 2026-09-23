"""
SatQuery AI -- Storage Management and Data Root Subsystem.
Handles:
- Dynamic data root resolution via SATQUERY_DATA_ROOT environment variable
- Storage capacity checks, space estimation, and safety guards
- Resumable download tracking and checksum verification
- Dataset-specific storage profiles (preventing reckless full extraction that overflows host disk)
"""

import os
import shutil
import hashlib
import json
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict

# Default root relative to project repository
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Storage profiles in bytes
DATASET_STORAGE_PROFILES = {
    "bigearthnet_txt_metadata": {
        "name": "BigEarthNet.txt (Parquet Metadata + Text)",
        "required_bytes": 500 * 1024 * 1024,  # ~500 MB
        "compressed_bytes": 466819745,
        "is_essential": True,
        "description": "Full 9.55M text and patch metadata index",
    },
    "bigearthnet_s1_s2_dev_subset": {
        "name": "BigEarthNet.txt (Selective Development Subset)",
        "required_bytes": 50 * 1024 * 1024,  # ~50 MB
        "compressed_bytes": 10 * 1024 * 1024,
        "is_essential": True,
        "description": "Representative paired S1/S2 patches for development and test validation",
    },
    "bigearthnet_s1_s2_full_archive": {
        "name": "BigEarthNet.txt (Full Raw S1 + S2 Image Archives)",
        "required_bytes": 350 * 1024 * 1024 * 1024,  # ~350 GB extracted
        "compressed_bytes": 160 * 1024 * 1024 * 1024,  # ~160 GB tar.zst
        "is_essential": False,
        "description": "Full raw archives for 464,044 Sentinel-1 and Sentinel-2 patches (Requires SATQUERY_DATA_ROOT mount)",
    },
    "vrsbench_annotations": {
        "name": "VRSBench (Evaluation Annotations)",
        "required_bytes": 150 * 1024 * 1024,  # ~150 MB
        "compressed_bytes": 102164307,
        "is_essential": True,
        "description": "Official public evaluation benchmark annotations (>62K samples)",
    },
    "vrsbench_images": {
        "name": "VRSBench (Evaluation Benchmark Imagery)",
        "required_bytes": 30 * 1024 * 1024 * 1024,  # ~30 GB
        "compressed_bytes": 20 * 1024 * 1024 * 1024,
        "is_essential": False,
        "description": "Source images for VRSBench evaluation (DOTA & DIOR imagery)",
    },
    "cdvqa_annotations": {
        "name": "CDVQA (Bitemporal Change VQA Annotations)",
        "required_bytes": 60 * 1024 * 1024,  # ~60 MB
        "compressed_bytes": 43402247,
        "is_essential": True,
        "description": "Official bitemporal question-answer pairs (122K QA pairs)",
    },
    "cdvqa_images": {
        "name": "CDVQA (SECOND Change Detection Imagery)",
        "required_bytes": 15 * 1024 * 1024 * 1024,  # ~15 GB
        "compressed_bytes": 8 * 1024 * 1024 * 1024,
        "is_essential": False,
        "description": "Aerial bitemporal images for SECOND dataset",
    },
}


@dataclass
class StorageCheckResult:
    """Outcome of storage feasibility verification."""
    target_path: str
    drive_mount: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    required_bytes: int
    deficit_bytes: int
    is_feasible: bool
    status: str  # "READY", "BLOCKED_STORAGE", "WARNING_LOW_DISK"
    notes: str


def get_data_root() -> Path:
    """
    Resolve the primary data root directory.
    Priority:
    1. SATQUERY_DATA_ROOT environment variable (e.g. D:\\SatQueryData or /data/satquery)
    2. Local repository data/ directory
    """
    env_root = os.getenv("SATQUERY_DATA_ROOT")
    if env_root and env_root.strip():
        resolved = Path(env_root.strip()).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved
    return DEFAULT_DATA_DIR.resolve()


def get_external_data_dir() -> Path:
    """Resolve directory for external datasets."""
    root = get_data_root()
    ext_dir = root / "external"
    ext_dir.mkdir(parents=True, exist_ok=True)
    return ext_dir


def get_manifests_dir() -> Path:
    """Resolve directory for dataset manifests."""
    root = get_data_root()
    m_dir = root / "manifests"
    m_dir.mkdir(parents=True, exist_ok=True)
    return m_dir


def check_storage_feasibility(
    dataset_key: str,
    target_path: Optional[Path | str] = None,
    safety_margin_gb: float = 10.0,
) -> StorageCheckResult:
    """
    Safely evaluate whether target drive has sufficient space for dataset extraction.
    Guards against reckless full extractions that overflow the operating system drive.
    """
    profile = DATASET_STORAGE_PROFILES.get(dataset_key)
    if not profile:
        raise ValueError(f"Unknown dataset key: {dataset_key}. Known keys: {list(DATASET_STORAGE_PROFILES.keys())}")

    target = Path(target_path) if target_path else get_data_root()
    target_str = str(target.resolve())

    # Get drive root / mountpoint
    try:
        total, used, free = shutil.disk_usage(target_str)
    except Exception as e:
        # Fallback for relative paths
        total, used, free = shutil.disk_usage(os.path.splitdrive(target_str)[0] or ".")

    drive = os.path.splitdrive(target_str)[0] or target_str
    req_bytes = profile["required_bytes"]
    safety_margin_bytes = int(safety_margin_gb * (1024 ** 3))
    effective_req_bytes = req_bytes + safety_margin_bytes

    deficit = max(0, effective_req_bytes - free)
    is_feasible = (free >= effective_req_bytes)

    if is_feasible:
        status = "READY"
        notes = f"Adequate storage available on {drive} ({free / (1024**3):.1f} GB free >= {effective_req_bytes / (1024**3):.1f} GB required with margin)."
    else:
        status = "BLOCKED_STORAGE"
        notes = (
            f"Insufficient space on drive {drive} for '{profile['name']}'. "
            f"Available: {free / (1024**3):.2f} GB | Required (with {safety_margin_gb}GB safety margin): {effective_req_bytes / (1024**3):.2f} GB | "
            f"Deficit: {deficit / (1024**3):.2f} GB. "
            f"Set SATQUERY_DATA_ROOT to an external high-capacity volume (e.g. D:\\SatQueryData or /mnt/data)."
        )

    return StorageCheckResult(
        target_path=target_str,
        drive_mount=drive,
        total_bytes=total,
        used_bytes=used,
        free_bytes=free,
        required_bytes=req_bytes,
        deficit_bytes=deficit,
        is_feasible=is_feasible,
        status=status,
        notes=notes,
    )


def compute_file_sha256(filepath: Path | str, chunk_size: int = 65536) -> str:
    """Compute cryptographic SHA-256 hash for checksum validation."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


class DownloadTracker:
    """Tracks resumable download states and integrity manifests."""

    def __init__(self, tracking_dir: Optional[Path] = None):
        self.tracking_dir = tracking_dir or (get_external_data_dir() / "download_state")
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

    def get_state_file(self, dataset_name: str) -> Path:
        safe_name = dataset_name.replace("/", "_").replace(" ", "_").lower()
        return self.tracking_dir / f"{safe_name}_state.json"

    def record_chunk(self, dataset_name: str, bytes_downloaded: int, total_expected: int, completed: bool = False, checksum: Optional[str] = None):
        state_file = self.get_state_file(dataset_name)
        data = {
            "dataset_name": dataset_name,
            "bytes_downloaded": bytes_downloaded,
            "total_expected": total_expected,
            "percent": round((bytes_downloaded / max(1, total_expected)) * 100, 2),
            "completed": completed,
            "checksum_sha256": checksum,
            "last_updated": str(Path.cwd()),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_state(self, dataset_name: str) -> Optional[dict[str, Any]]:
        state_file = self.get_state_file(dataset_name)
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
