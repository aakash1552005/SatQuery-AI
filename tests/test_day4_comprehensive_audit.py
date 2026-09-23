"""
SatQuery AI -- Day 4 Comprehensive Audit & Real-Data Validation Tests.
Covers:
- SATQUERY_DATA_ROOT external root configuration
- Storage feasibility calculations and safety margins
- BigEarthNet.txt stratified subset manifests (train, val, test)
- Task balance across binary, mcq, bounding box, and captioning
- Radiometric fidelity in multimodal preprocessing (SAR linear ratio, optical stats)
- Quality control and zero cross-split leakage enforcement
- Zero fake checkpoint and zero fake metric honesty guarantees
"""

import os
import json
import pytest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

from src.data.storage_manager import (
    get_data_root,
    check_storage_feasibility,
    DATASET_STORAGE_PROFILES,
    compute_file_sha256,
)
from src.data.preprocessing import MultimodalRSPreprocessor
from src.adaptation.lora_config import load_lora_config
from src.adaptation.training_preflight import check_training_feasibility
from scripts.run_day4_full import step_preflight, step_prepare_data


def test_satquery_data_root_environment_override(monkeypatch, tmp_path):
    """Verify that SATQUERY_DATA_ROOT environment variable dynamically redirects data resolution."""
    custom_dir = tmp_path / "custom_satquery_data"
    monkeypatch.setenv("SATQUERY_DATA_ROOT", str(custom_dir))

    resolved_root = get_data_root()
    assert resolved_root == custom_dir
    assert resolved_root.exists()


def test_storage_feasibility_check():
    """Verify storage estimation logic correctly evaluates capacity."""
    # Metadata should be feasible on host
    meta_check = check_storage_feasibility("bigearthnet_txt_metadata")
    assert meta_check.required_bytes < 1 * (1024 ** 3)
    assert meta_check.is_feasible is True

    # Full raw archive (350+ GB) should trigger safety guard or blocked status on 100GB disk
    full_check = check_storage_feasibility("bigearthnet_s1_s2_full_archive")
    assert full_check.required_bytes >= 300 * (1024 ** 3)
    if full_check.free_bytes < (350 + 10) * (1024 ** 3):
        assert full_check.status == "BLOCKED_STORAGE"
        assert full_check.is_feasible is False
        assert full_check.deficit_bytes > 0


def test_real_data_subset_manifests_exist():
    """Verify that deterministic subset manifests exist and have correct schema."""
    manifests_dir = ROOT / "data" / "manifests"
    for fname in ["ben_train_subset.json", "ben_val_subset.json", "ben_test_subset.json"]:
        p = manifests_dir / fname
        assert p.exists(), f"Subset manifest {fname} missing"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["dataset_name"] == "BigEarthNet.txt"
        assert len(data["samples"]) > 0
        assert "provenance" in data


def test_subset_manifest_task_balance():
    """Verify stratified balance across binary, mcq, bounding box, and captioning tasks."""
    train_manifest = ROOT / "data" / "manifests" / "ben_train_subset.json"
    with open(train_manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    dist = data.get("task_distribution", {})
    assert "binary" in dist
    assert "mcq" in dist
    assert "bounding box" in dist
    assert "captioning" in dist

    # All 4 tasks should be represented in training subset
    assert dist["binary"] == 250
    assert dist["mcq"] == 250
    assert dist["bounding box"] == 250
    assert dist["captioning"] == 250


def test_zero_leakage_between_sampled_subsets():
    """Verify strict isolation and zero record overlap between train, val, and test subsets."""
    manifests_dir = ROOT / "data" / "manifests"
    with open(manifests_dir / "ben_train_subset.json", "r", encoding="utf-8") as f:
        train_ids = {s["record_id"] for s in json.load(f)["samples"]}
    with open(manifests_dir / "ben_val_subset.json", "r", encoding="utf-8") as f:
        val_ids = {s["record_id"] for s in json.load(f)["samples"]}
    with open(manifests_dir / "ben_test_subset.json", "r", encoding="utf-8") as f:
        test_ids = {s["record_id"] for s in json.load(f)["samples"]}

    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val records overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test records overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test records overlap!"


def test_s1_s2_patch_alignment_in_subsets():
    """Verify Sentinel-1 and Sentinel-2 patch tiles correspond to the same footprint."""
    manifests_dir = ROOT / "data" / "manifests"
    with open(manifests_dir / "ben_train_subset.json", "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    for s in samples[:50]:  # check first 50 samples
        s1 = s["s1_name"]
        s2 = s["patch_id"]
        s1_parts = s1.split("_")
        s2_parts = s2.split("_")
        # Grid coordinates (e.g. 7_53 vs 07_53) must match
        assert int(s1_parts[-2]) == int(s2_parts[-2]) and int(s1_parts[-1]) == int(s2_parts[-1]), (
            f"Footprint mismatch: S1 {s1_parts[-2]}_{s1_parts[-1]} != S2 {s2_parts[-2]}_{s2_parts[-1]}"
        )


def test_preprocessor_radiometric_fidelity():
    """Verify preprocessing preserves 16-bit optical reflectance and computes valid SAR ratios."""
    preprocessor = MultimodalRSPreprocessor()

    # Synthetic optical 16-bit stack (4 bands: Blue, Green, Red, NIR)
    raw_optical = np.array([
        [[400, 500], [600, 700]],
        [[550, 650], [750, 850]],
        [[500, 600], [700, 800]],
        [[2000, 2200], [2400, 2600]],
    ], dtype=np.uint16)

    stats = preprocessor.compute_optical_stats(raw_optical)
    assert "B02_Blue" in stats
    assert "B08_NIR" in stats
    assert stats["B08_NIR"]["mean"] > stats["B04_Red"]["mean"]

    scaled_optical = preprocessor.preprocess_optical(raw_optical)
    assert scaled_optical.shape == (4, 2, 2)
    assert np.all(scaled_optical >= 0.0) and np.all(scaled_optical <= 1.0)
    # 2600 / 10000 = 0.26
    assert np.isclose(scaled_optical[3, 1, 1], 0.26, atol=1e-4)

    # SAR ratio tests
    vv_db = np.array([-10.0])
    vh_db = np.array([-15.0])
    sar_feats = preprocessor.compute_sar_polarization_features(vv_db, vh_db)
    # Expected linear ratio = 10^((-10 - (-15))/10) = 10^0.5 ≈ 3.162277
    assert np.isclose(sar_feats["linear_ratio"][0], 10.0 ** 0.5, rtol=1e-4)
    # Expected dB difference = -10 - (-15) = 5.0
    assert np.isclose(sar_feats["difference_db"][0], 5.0, atol=1e-4)


def test_model_honesty_and_remote_package_creation():
    """Verify that model inventory honestly declares zero trained checkpoints and packages remote runner."""
    inv_path = ROOT / "docs" / "model_inventory.json"
    with open(inv_path, "r", encoding="utf-8") as f:
        inv = json.load(f)

    assert inv["overall_ml_status"] == "NO_SATQUERY_TRAINED_MODEL_EXISTS"
    for m in inv["models"]:
        assert m["checkpoint"] == "NONE"
        assert m["metrics_available"] is False

    pkg_dir = ROOT / "artifacts" / "day4_remote_training_package"
    assert pkg_dir.exists()
    assert (pkg_dir / "bigearthnet_txt_lora.yaml").exists()
    assert (pkg_dir / "requirements.txt").exists()
    assert (pkg_dir / "launch_remote_training.sh").exists()
