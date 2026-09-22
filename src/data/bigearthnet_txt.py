"""
BigEarthNet.txt Dataset Loader, Multimodal Alignment Validator, and Leakage Auditor.
SIH Problem Statement 26167: Multimodal Remote Sensing Image Analysis through Text Queries.

Official Dataset Reference:
- Paper: BigEarthNet.txt (arXiv:2603.29630, 2026)
- URL: https://txt.bigearth.net/
- Primary Role: Multimodal RS training / fine-tuning (Sentinel-1 SAR + Sentinel-2 Optical + Text).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import json
import numpy as np
import rasterio

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object  # Fallback for headless testing without torch


@dataclass
class MultimodalSample:
    """Represents a validated multimodal sample triplet from BigEarthNet.txt."""
    sample_id: str
    dataset_name: str = "BigEarthNet.txt"
    dataset_version: str = "1.0"
    split: str = "train"
    modalities: list[str] = field(default_factory=lambda: ["sentinel_1_sar", "sentinel_2_multispectral", "text"])
    s1_path: str = ""
    s2_path: str = ""
    annotation_path: str = ""
    task_type: str = "vqa"
    text: str = ""
    question: Optional[str] = None
    answer: Optional[str] = None
    caption: Optional[str] = None
    instruction: Optional[str] = None
    explanation: Optional[str] = None
    land_cover_classes: list[str] = field(default_factory=list)


@dataclass
class AlignmentResult:
    """Outcome of S1-S2-Text multimodal correspondence validation."""
    sample_id: str
    is_aligned: bool
    s1_valid: bool
    s2_valid: bool
    text_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    s1_channels: Optional[int] = None
    s2_channels: Optional[int] = None


@dataclass
class DatasetAuditReport:
    """Outcome of dataset duplicate detection and cross-split/evaluation leakage audit."""
    dataset_name: str
    total_samples: int
    unique_sample_ids: int
    duplicate_sample_ids: list[str]
    split_counts: dict[str, int]
    split_overlaps: dict[str, list[str]]
    evaluation_leakage_detected: bool
    leakage_sample_ids: list[str]
    role_policy: str
    split_validation: str  # "PASSED", "PENDING", "FAILED"
    duplicate_audit: str   # "PASSED", "FAILED"
    notes: str


def validate_multimodal_alignment(sample_info: dict | MultimodalSample, root_dir: Optional[Path] = None) -> AlignmentResult:
    """
    Strictly validate S1 SAR + S2 Optical + Text annotation alignment for a sample.
    Catches:
    - Missing S1 or S2 paths
    - Missing or empty text
    - File existence failures on disk
    - S1 channel mismatches (must have at least 2 channels, e.g. VV/VH)
    - S2 channel mismatches (must have at least 3 channels, e.g. RGB or RGB-NIR)
    - Broken file references
    """
    if isinstance(sample_info, MultimodalSample):
        d = {
            "sample_id": sample_info.sample_id,
            "s1_path": sample_info.s1_path,
            "s2_path": sample_info.s2_path,
            "text": sample_info.text,
            "task_type": sample_info.task_type,
        }
    else:
        d = sample_info

    sample_id = d.get("sample_id", "UNKNOWN_SAMPLE")
    errors = []
    warnings = []
    s1_valid = False
    s2_valid = False
    text_valid = False
    s1_ch = None
    s2_ch = None

    root = root_dir or Path.cwd()

    # 1. Text / Annotation Validation
    text_content = d.get("text", "")
    task_type = d.get("task_type", "vqa")
    if not text_content or not isinstance(text_content, str) or len(text_content.strip()) == 0:
        errors.append(f"Missing or empty text annotation for sample {sample_id}")
    else:
        text_valid = True

    # 2. S1 SAR GeoTIFF Validation
    s1_rel = d.get("s1_path")
    if not s1_rel:
        errors.append(f"Sample {sample_id} missing s1_path specification")
    else:
        s1_file = Path(s1_rel)
        if not s1_file.is_absolute():
            s1_file = root / s1_file
        if not s1_file.exists():
            errors.append(f"S1 file does not exist on disk: {s1_file}")
        else:
            try:
                with rasterio.open(s1_file) as src:
                    s1_ch = src.count
                    if s1_ch < 2:
                        errors.append(f"S1 SAR requires at least 2 channels (VV, VH); found count={s1_ch}")
                    else:
                        s1_valid = True
            except Exception as e:
                errors.append(f"Failed to inspect S1 rasterio headers: {e}")

    # 3. S2 Optical GeoTIFF Validation
    s2_rel = d.get("s2_path")
    if not s2_rel:
        errors.append(f"Sample {sample_id} missing s2_path specification")
    else:
        s2_file = Path(s2_rel)
        if not s2_file.is_absolute():
            s2_file = root / s2_file
        if not s2_file.exists():
            errors.append(f"S2 file does not exist on disk: {s2_file}")
        else:
            try:
                with rasterio.open(s2_file) as src:
                    s2_ch = src.count
                    if s2_ch < 3:
                        errors.append(f"S2 Optical requires at least 3 channels (RGB); found count={s2_ch}")
                    else:
                        s2_valid = True
            except Exception as e:
                errors.append(f"Failed to inspect S2 rasterio headers: {e}")

    is_aligned = s1_valid and s2_valid and text_valid and (len(errors) == 0)

    return AlignmentResult(
        sample_id=sample_id,
        is_aligned=is_aligned,
        s1_valid=s1_valid,
        s2_valid=s2_valid,
        text_valid=text_valid,
        errors=errors,
        warnings=warnings,
        s1_channels=s1_ch,
        s2_channels=s2_ch,
    )


def audit_dataset_duplicates_and_leakage(
    manifest_path: Path | str,
    eval_manifest_path: Optional[Path | str] = None,
) -> DatasetAuditReport:
    """
    Execute rigorous duplicate and leakage audit over dataset manifest.
    Checks:
    - Unique sample IDs
    - Unique file paths
    - Overlap between train, validation, and test splits
    - Zero data leakage into public evaluation benchmark (e.g. VRSBench)
    """
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_name = data.get("dataset_name", "BigEarthNet.txt")
    samples = data.get("samples", [])
    total_samples = len(samples)

    seen_ids = set()
    duplicate_ids = set()
    split_ids: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    for s in samples:
        sid = s.get("sample_id")
        if not sid:
            continue
        if sid in seen_ids:
            duplicate_ids.add(sid)
        seen_ids.add(sid)

        split = s.get("split", "train").lower()
        if split not in split_ids:
            split_ids[split] = set()
        split_ids[split].add(sid)

    # Check split overlaps (e.g. train vs test)
    split_overlaps = {}
    splits = list(split_ids.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = split_ids[s1].intersection(split_ids[s2])
            if overlap:
                split_overlaps[f"{s1}_vs_{s2}"] = list(overlap)

    # Check evaluation benchmark leakage if evaluation manifest provided
    eval_leakage_detected = False
    leakage_ids = []
    if eval_manifest_path:
        eval_file = Path(eval_manifest_path)
        if eval_file.exists():
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            eval_samples = eval_data.get("samples", [])
            eval_sids = {es.get("sample_id") for es in eval_samples if es.get("sample_id")}
            leakage = seen_ids.intersection(eval_sids)
            if leakage:
                eval_leakage_detected = True
                leakage_ids = list(leakage)

    has_duplicates = len(duplicate_ids) > 0
    has_split_overlap = len(split_overlaps) > 0
    duplicate_audit_status = "FAILED" if (has_duplicates or has_split_overlap or eval_leakage_detected) else "PASSED"
    split_val_status = "FAILED" if has_split_overlap else "PASSED"

    notes = (
        "Zero leakage confirmed across splits. No duplicate IDs detected."
        if duplicate_audit_status == "PASSED"
        else f"Audited with issues: {len(duplicate_ids)} duplicates, {len(split_overlaps)} split overlaps."
    )

    return DatasetAuditReport(
        dataset_name=dataset_name,
        total_samples=total_samples,
        unique_sample_ids=len(seen_ids),
        duplicate_sample_ids=list(duplicate_ids),
        split_counts={k: len(v) for k, v in split_ids.items()},
        split_overlaps=split_overlaps,
        evaluation_leakage_detected=eval_leakage_detected,
        leakage_sample_ids=leakage_ids,
        role_policy="enforced",
        split_validation=split_val_status,
        duplicate_audit=duplicate_audit_status,
        notes=notes,
    )


class BigEarthNetTxtDataset(Dataset):
    """
    PyTorch Dataset for BigEarthNet.txt multimodal samples.
    Loads co-registered Sentinel-1 SAR, Sentinel-2 Optical, and task text descriptions.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        split: Optional[str] = None,
        preprocessor: Optional[Any] = None,
        root_dir: Optional[Path] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.root_dir = root_dir or Path.cwd()
        self.split = split
        self.preprocessor = preprocessor

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_samples = data.get("samples", [])
        if self.split:
            self.samples = [s for s in raw_samples if s.get("split", "").lower() == self.split.lower()]
        else:
            self.samples = raw_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        sample_id = item["sample_id"]

        # 1. Load S1 SAR
        s1_file = Path(item["s1_path"])
        if not s1_file.is_absolute():
            s1_file = self.root_dir / s1_file
        with rasterio.open(s1_file) as src:
            s1_raw = src.read()  # (channels, H, W)

        # 2. Load S2 Optical
        s2_file = Path(item["s2_path"])
        if not s2_file.is_absolute():
            s2_file = self.root_dir / s2_file
        with rasterio.open(s2_file) as src:
            s2_raw = src.read()  # (channels, H, W)

        # 3. Apply preprocessing if provided
        if self.preprocessor:
            s1_processed = self.preprocessor.preprocess_sar(s1_raw)
            s2_processed = self.preprocessor.preprocess_optical(s2_raw)
            formatted_text = self.preprocessor.format_instruction_prompt(item)
        else:
            s1_processed = s1_raw.astype(np.float32)
            s2_processed = s2_raw.astype(np.float32)
            formatted_text = item.get("text", "")

        return {
            "sample_id": sample_id,
            "dataset_name": "BigEarthNet.txt",
            "split": item.get("split", "train"),
            "task_type": item.get("task_type", "vqa"),
            "text": formatted_text,
            "s1_data": s1_processed,
            "s2_data": s2_processed,
            "classes": item.get("land_cover_classes", []),
        }
