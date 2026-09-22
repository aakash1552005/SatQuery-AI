"""
=============================================================================
SATQUERY AI -- REAL DATASET ACQUISITION & VALIDATION TESTS (PHASE 28)
=============================================================================
Validates:
  - test_real_dataset_manifest
  - test_real_bigearthnet_record_validation
  - test_real_s1_s2_alignment
  - test_real_bigearthnet_split_validation
  - test_real_bigearthnet_duplicate_audit
  - test_vrsbench_manifest
  - test_vrsbench_coordinate_format
  - test_cdvqa_manifest
  - test_dataset_inventory
  - test_synthetic_real_separation
=============================================================================
"""

import json
from pathlib import Path
import pytest
import pyarrow.parquet as pq

from src.data.bigearthnet_txt import validate_real_ben_txt_alignment
from src.data.vrsbench import normalize_grounding_coordinates, VRSBenchManifest
from src.data.cdvqa import CDVQAManifest

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "data" / "manifests"
EXT_DIR = ROOT / "data" / "external"
BEN_PARQUET = EXT_DIR / "bigearthnet_txt" / "metadata" / "BigEarthNet.txt.parquet"


def test_real_dataset_manifest():
    report_file = MANIFESTS / "bigearthnet_txt_dataset_report.json"
    assert report_file.exists()
    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["dataset_name"] == "BigEarthNet.txt"
    assert data["record_count"] == 9553962
    assert data["unique_sample_count"] == 464044
    assert data["license"] == "CDLA-Permissive-1.0"
    assert "arxiv.org/abs/2603.29630" in data["official_source"]
    assert data["validation_status"] == "VALIDATED"


def test_real_bigearthnet_record_validation():
    assert BEN_PARQUET.exists()
    table = pq.read_table(BEN_PARQUET).slice(0, 50)
    assert table.num_rows == 50
    df = table.to_pandas()
    assert "ID" in df.columns
    assert "s1_name" in df.columns
    assert "patch_id" in df.columns
    assert "input" in df.columns
    assert "output" in df.columns
    assert "split" in df.columns

    row0 = df.iloc[0]
    assert row0["s1_name"].startswith("S1")
    assert row0["patch_id"].startswith("S2")
    assert len(row0["input"]) > 0
    assert len(row0["output"]) > 0


def test_real_s1_s2_alignment():
    result = validate_real_ben_txt_alignment(BEN_PARQUET, max_samples=200)
    assert result["status"] == "VERIFIED"
    assert result["valid_pairs"] == 200
    assert result["identity_mismatch"] == 0
    assert result["missing_s1"] == 0
    assert result["missing_s2"] == 0


def test_real_bigearthnet_split_validation():
    split_file = MANIFESTS / "bigearthnet_txt_split_report.json"
    assert split_file.exists()
    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "VERIFIED"
    assert "official_full_release" in data
    assert data["official_full_release"]["train_count"] == 4674281
    assert data["official_full_release"]["validation_count"] == 2454690
    assert data["official_full_release"]["test_count"] == 2409962
    assert data["official_full_release"]["bench_count"] == 15029
    assert data["official_full_release"]["total_count"] == 9553962


def test_real_bigearthnet_duplicate_audit():
    leakage_file = MANIFESTS / "bigearthnet_txt_leakage_report.json"
    assert leakage_file.exists()
    with open(leakage_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "PASSED"
    assert data["duplicate_id_count"] == 0
    assert data["train_val_overlap"] == 0
    assert data["train_test_overlap"] == 0
    assert data["val_test_overlap"] == 0
    assert data["train_bench_overlap"] == 0


def test_vrsbench_manifest():
    vrs_file = MANIFESTS / "vrsbench_manifest.json"
    assert vrs_file.exists()
    with open(vrs_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["dataset_name"] == "VRSBench"
    assert data["role"] == "public_evaluation"
    assert data["training_allowed"] is False
    assert data["evaluation_allowed"] is True
    assert data["total_eval_samples"] > 30000

    parser = VRSBenchManifest(EXT_DIR / "vrsbench" / "annotations")
    assert parser.validate_evaluation_policy() is True


def test_vrsbench_coordinate_format():
    token_str = "{<20><30><40><50>}"
    res = normalize_grounding_coordinates(token_str, target_format="normalized_float")
    assert res["source_coordinate_format"] == "token_0_100"
    assert res["target_coordinate_format"] == "normalized_float"
    assert res["converted"] == [0.2, 0.3, 0.4, 0.5]

    px_res = normalize_grounding_coordinates(token_str, target_format="pixel_xyxy", image_width=1000, image_height=1000)
    assert px_res["converted"] == [300, 200, 500, 400]

    corner_coords = [0.1, 0.2, 0.3, 0.2, 0.3, 0.6, 0.1, 0.6]
    c_res = normalize_grounding_coordinates(corner_coords, target_format="normalized_float")
    assert c_res["source_coordinate_format"] == "obj_corner"
    assert c_res["converted"] == [0.2, 0.1, 0.6, 0.3]


def test_cdvqa_manifest():
    cdvqa_file = MANIFESTS / "cdvqa_manifest.json"
    assert cdvqa_file.exists()
    with open(cdvqa_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["dataset_name"] == "CDVQA"
    assert data["role"] == "temporal_vqa_validation"
    assert data["training_allowed"] is False
    assert data["evaluation_allowed"] is True
    assert data["annotations_downloaded"] is True
    assert data["images_downloaded"] is False
    assert data["splits"]["train"]["question_count"] > 60000

    validator = CDVQAManifest(EXT_DIR / "cdvqa")
    struct = validator.validate_dataset_structure()
    assert struct["status"] == "METADATA_READY"


def test_dataset_inventory():
    inv_file = MANIFESTS / "dataset_inventory.json"
    assert inv_file.exists()
    with open(inv_file, "r", encoding="utf-8") as f:
        inv = json.load(f)
    assert "storage_audit" in inv
    assert "datasets" in inv
    ds = inv["datasets"]
    assert "bigearthnet_txt" in ds
    assert "vrsbench" in ds
    assert "cdvqa" in ds
    assert "synthetic_engineering" in ds
    assert ds["bigearthnet_txt"]["role"] == "training_finetuning"
    assert ds["vrsbench"]["role"] == "public_evaluation"
    assert ds["cdvqa"]["role"] == "temporal_vqa_validation"


def test_synthetic_real_separation():
    synth_dir = ROOT / "data" / "samples"
    assert synth_dir.exists()
    synth_files = list(synth_dir.rglob("*.tif"))
    assert len(synth_files) > 0, "Synthetic rasters must exist for engineering tests"

    # Real data directory
    assert (EXT_DIR / "bigearthnet_txt" / "metadata" / "BigEarthNet.txt.parquet").exists()

    with open(MANIFESTS / "bigearthnet_txt_dataset_report.json", "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["record_count"] == 9553962
    assert "synthetic" not in rep["dataset_name"].lower()
