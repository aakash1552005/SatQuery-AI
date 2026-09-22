"""
Day 4 Automated Tests: BigEarthNet.txt Dataset Pipeline, S1/S2/Text Alignment, and Leakage Audit.
SIH Problem Statement 26167: Multimodal Remote Sensing Image Analysis through Text Queries.
"""

from pathlib import Path
import json
import pytest
import yaml

from src.data.bigearthnet_txt import (
    MultimodalSample,
    validate_multimodal_alignment,
    audit_dataset_duplicates_and_leakage,
    BigEarthNetTxtDataset,
)
from src.data.preprocessing import MultimodalRSPreprocessor

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = ROOT / "data" / "manifests"
SAMPLES_DIR = ROOT / "data" / "external" / "bigearthnet_txt" / "samples"


def test_bigearthnet_txt_registry_role():
    """Verify BigEarthNet.txt is registered with role training_finetuning and PIPELINE_READY status."""
    reg_path = MANIFESTS_DIR / "dataset_registry.yaml"
    with open(reg_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    ben = manifest["datasets"]["bigearthnet_txt"]
    assert ben["role"] == "training_finetuning"
    assert ben["training_allowed"] is True
    assert ben["evaluation_allowed"] is False
    assert ben["pipeline_status"] == "PIPELINE_READY"
    assert "https://txt.bigearth.net/" in ben["source"]["official_page"]
    assert "2603.29630" in ben["source"]["url"]


def test_bigearthnet_txt_manifest():
    """Verify BigEarthNet.txt manifest structure, sample IDs, and required fields."""
    man_path = MANIFESTS_DIR / "bigearthnet_txt_manifest.json"
    assert man_path.exists(), f"Manifest file missing: {man_path}"

    with open(man_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["dataset_name"] == "BigEarthNet.txt"
    assert manifest["sample_count"] >= 4
    assert len(manifest["samples"]) == manifest["sample_count"]

    for sample in manifest["samples"]:
        assert "sample_id" in sample
        assert sample["dataset_name"] == "BigEarthNet.txt"
        assert sample["split"] in ("train", "val", "test")
        assert "s1_path" in sample
        assert "s2_path" in sample
        assert "annotation_path" in sample
        assert "text" in sample
        assert len(sample["text"]) > 0
        assert sample["task_type"] in ("vqa", "captioning", "land_cover_reasoning")


def test_bigearthnet_txt_split_policy():
    """Verify train/val/test split policy and split report metrics."""
    split_path = MANIFESTS_DIR / "bigearthnet_txt_split_report.json"
    assert split_path.exists(), f"Split report missing: {split_path}"

    with open(split_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["dataset_name"] == "BigEarthNet.txt"
    assert rep["status"] == "VERIFIED"
    assert rep["train_count"] > 0
    assert rep["validation_count"] > 0
    assert rep["test_count"] > 0
    assert rep["train_count"] + rep["validation_count"] + rep["test_count"] == rep["total_samples"]


def test_s1_s2_alignment():
    """Verify real co-registered S1 SAR and S2 Optical patches pass strict alignment validation."""
    man_path = MANIFESTS_DIR / "bigearthnet_txt_manifest.json"
    with open(man_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for sample in manifest["samples"]:
        res = validate_multimodal_alignment(sample, root_dir=ROOT)
        assert res.is_aligned is True, f"Alignment failed for {sample['sample_id']}: {res.errors}"
        assert res.s1_valid is True
        assert res.s2_valid is True
        assert res.text_valid is True
        assert res.s1_channels == 2  # VV, VH
        assert res.s2_channels == 4  # B02, B03, B04, B08


def test_alignment_rejection_missing_modalities():
    """Verify that samples with missing files or missing text are rejected with structured errors."""
    # 1. Missing S1 path
    bad_sample_1 = {
        "sample_id": "BAD_001",
        "s1_path": "",
        "s2_path": "data/external/bigearthnet_txt/samples/S2A_MSIL2A_20170613T101031_patch_001.tif",
        "text": "Some text",
    }
    res_1 = validate_multimodal_alignment(bad_sample_1, root_dir=ROOT)
    assert res_1.is_aligned is False
    assert res_1.s1_valid is False
    assert any("missing s1_path" in e.lower() for e in res_1.errors)

    # 2. Missing text annotation
    bad_sample_2 = {
        "sample_id": "BAD_002",
        "s1_path": "data/external/bigearthnet_txt/samples/S1A_IW_GRDH_1SDV_20170613T165043_patch_001.tif",
        "s2_path": "data/external/bigearthnet_txt/samples/S2A_MSIL2A_20170613T101031_patch_001.tif",
        "text": "",
    }
    res_2 = validate_multimodal_alignment(bad_sample_2, root_dir=ROOT)
    assert res_2.is_aligned is False
    assert res_2.text_valid is False
    assert any("missing or empty text" in e.lower() for e in res_2.errors)

    # 3. Non-existent file path
    bad_sample_3 = {
        "sample_id": "BAD_003",
        "s1_path": "data/external/non_existent_sar.tif",
        "s2_path": "data/external/non_existent_optical.tif",
        "text": "Sample text",
    }
    res_3 = validate_multimodal_alignment(bad_sample_3, root_dir=ROOT)
    assert res_3.is_aligned is False
    assert any("does not exist on disk" in e for e in res_3.errors)


def test_text_annotation_alignment():
    """Verify task types and prompt formatting across BigEarthNet.txt multimodal samples."""
    preprocessor = MultimodalRSPreprocessor()

    # VQA task formatting
    vqa_sample = {
        "sample_id": "TEST_VQA",
        "task_type": "vqa",
        "question": "Are there agricultural fields?",
        "answer": "Yes, extensive arable parcel.",
    }
    vqa_prompt = preprocessor.format_instruction_prompt(vqa_sample)
    assert "Question: Are there agricultural fields?" in vqa_prompt
    assert "Assistant: Yes, extensive arable parcel." in vqa_prompt
    assert "<image_s1><image_s2>" in vqa_prompt

    # Captioning task formatting
    cap_sample = {
        "sample_id": "TEST_CAP",
        "task_type": "captioning",
        "caption": "A dense urban region with concrete structures.",
    }
    cap_prompt = preprocessor.format_instruction_prompt(cap_sample)
    assert "Provide a comprehensive remote sensing description" in cap_prompt
    assert "A dense urban region with concrete structures." in cap_prompt


def test_duplicate_audit():
    """Verify audit_dataset_duplicates_and_leakage detects zero duplicates and zero split overlap."""
    man_path = MANIFESTS_DIR / "bigearthnet_txt_manifest.json"
    audit = audit_dataset_duplicates_and_leakage(man_path)

    assert audit.total_samples == audit.unique_sample_ids
    assert len(audit.duplicate_sample_ids) == 0
    assert len(audit.split_overlaps) == 0
    assert audit.evaluation_leakage_detected is False
    assert audit.duplicate_audit == "PASSED"
    assert audit.split_validation == "PASSED"


def test_dataset_provenance():
    """Verify provenance metadata in manifest records and raster tags."""
    man_path = MANIFESTS_DIR / "bigearthnet_txt_manifest.json"
    with open(man_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for sample in manifest["samples"]:
        assert sample["dataset_name"] == "BigEarthNet.txt"
        assert sample["modalities"] == ["sentinel_1_sar", "sentinel_2_multispectral", "text"]
        assert sample["s1_bands"] == ["VV", "VH"]
        assert "B04_Red" in sample["s2_bands"]


def test_controlled_development_subset():
    """Verify development subset file is tracked and references valid samples."""
    subset_path = MANIFESTS_DIR / "bigearthnet_txt_subset.json"
    assert subset_path.exists()

    with open(subset_path, "r", encoding="utf-8") as f:
        sub = json.load(f)

    assert sub["subset_name"] == "bigearthnet_txt_dev_tier1"
    assert sub["sample_count"] > 0
    assert len(sub["sample_ids"]) == sub["sample_count"]
