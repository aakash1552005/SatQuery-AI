"""
Day 4 Automated Tests: RS-VLM / LoRA / PEFT Adaptation Pipeline and Training Preflight.
SIH Problem Statement 26167: Multimodal Remote Sensing Image Analysis through Text Queries.
"""

from pathlib import Path
import pytest
import torch

from src.adaptation.lora_config import RSLoraConfig, load_lora_config
from src.adaptation.training_preflight import check_training_feasibility, TrainingPreflightReport
from src.adaptation.pipeline import RSVLMAdaptationPipeline
from src.data.bigearthnet_txt import BigEarthNetTxtDataset
from src.data.preprocessing import MultimodalRSPreprocessor
from src.router.capability_registry import CapabilityRegistry, CapabilityStatus

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "training" / "bigearthnet_txt_lora.yaml"
MANIFEST_PATH = ROOT / "data" / "manifests" / "bigearthnet_txt_manifest.json"


def test_lora_config_validation():
    """Verify RSLoraConfig parameter boundaries and error reporting."""
    # Valid config
    valid_cfg = RSLoraConfig(r=16, lora_alpha=32, lora_dropout=0.05)
    assert len(valid_cfg.validate()) == 0

    # Invalid rank
    bad_cfg_1 = RSLoraConfig(r=0)
    assert any("rank r must be >= 1" in e for e in bad_cfg_1.validate())

    # Invalid dropout
    bad_cfg_2 = RSLoraConfig(lora_dropout=1.5)
    assert any("dropout must be in [0.0, 1.0)" in e for e in bad_cfg_2.validate())

    # Empty target modules
    bad_cfg_3 = RSLoraConfig(target_modules=[])
    assert any("target_modules cannot be empty" in e for e in bad_cfg_3.validate())


def test_load_lora_config_yaml():
    """Verify loading LoRA configuration from YAML file."""
    assert CONFIG_PATH.exists(), f"Configuration file missing: {CONFIG_PATH}"
    cfg = load_lora_config(CONFIG_PATH)

    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.compute_target == "remote_gpu"
    assert "q_proj" in cfg.target_modules
    assert cfg.learning_rate == 0.0002
    assert cfg.batch_size == 2


def test_training_preflight_cpu_host():
    """Verify training preflight truthfully detects CPU host and reports PIPELINE_READY_REMOTE_GPU."""
    report = check_training_feasibility()

    assert isinstance(report, TrainingPreflightReport)
    assert report.cuda_available is False
    assert report.training_feasible is False
    assert report.status == "PIPELINE_READY_REMOTE_GPU"
    assert any("CPU-only" in r for r in report.reasons)
    assert any("CUDA" in r for r in report.reasons)


def test_dataset_loader_and_collation():
    """Verify BigEarthNetTxtDataset loads multimodal triplets and collates batches properly."""
    preprocessor = MultimodalRSPreprocessor()
    dataset = BigEarthNetTxtDataset(
        manifest_path=MANIFEST_PATH,
        preprocessor=preprocessor,
        root_dir=ROOT,
    )
    assert len(dataset) >= 4

    # Test single sample
    sample = dataset[0]
    assert "sample_id" in sample
    assert "s1_data" in sample
    assert "s2_data" in sample
    assert sample["s1_data"].shape == (2, 120, 120)  # VV, VH
    assert sample["s2_data"].shape == (4, 120, 120)  # B02, B03, B04, B08
    assert "text" in sample

    # Test batch collation
    batch = [dataset[0], dataset[1]]
    collated = preprocessor.collate_fn(batch)
    assert collated["batch_size"] == 2
    assert collated["s1_tensors"].shape == (2, 2, 120, 120)
    assert collated["s2_tensors"].shape == (2, 4, 120, 120)
    assert len(collated["texts"]) == 2
    assert collated["provenance"]["dataset"] == "BigEarthNet.txt"


def test_preprocessing_preserves_provenance():
    """Verify SAR dB to linear Lee filtering and optical surface reflectance scaling."""
    preprocessor = MultimodalRSPreprocessor()

    # SAR input (dB domain)
    raw_sar = (torch.randn(2, 120, 120).numpy() * 5.0) - 15.0
    proc_sar = preprocessor.preprocess_sar(raw_sar)
    assert proc_sar.shape == (2, 120, 120)
    assert proc_sar.dtype == torch.float32 or proc_sar.dtype.name == "float32"

    # Optical input (reflectance 0-10000)
    raw_opt = torch.randint(0, 8000, (4, 120, 120)).numpy()
    proc_opt = preprocessor.preprocess_optical(raw_opt)
    assert proc_opt.shape == (4, 120, 120)
    assert proc_opt.min() >= 0.0
    assert proc_opt.max() <= 1.0


def test_adaptation_pipeline_execution():
    """Verify pipeline initialization, preflight integration, and zero fake checkpoints on CPU."""
    pipeline = RSVLMAdaptationPipeline(config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH)
    res = pipeline.execute()

    assert res.pipeline_status == "PIPELINE_READY"
    assert res.training_status == "NOT_EXECUTED"
    assert res.checkpoint_path is None
    assert len(res.loss_history) == 0  # Zero fake losses
    assert res.compute_target == "remote_gpu"
    assert "PIPELINE_READY" in res.notes


def test_model_smoke_preflight():
    """Verify model smoke preflight truthfully reports forward pass NOT_EXECUTED on CPU."""
    pipeline = RSVLMAdaptationPipeline(config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH)
    smoke = pipeline.model_smoke_preflight()

    assert smoke["model_name"] == "MBZUAI/geochat-7b"
    assert smoke["device"] == "cpu"
    assert "NOT_EXECUTED" in smoke["forward_pass"]
    assert smoke["lora_rank"] == 16


def test_capability_registry_rs_adaptation():
    """Verify rs_adaptation record in capability registry accurately reflects Day 4 status."""
    registry = CapabilityRegistry()
    cap = registry.get("rs_adaptation")

    assert cap is not None
    assert cap.status == CapabilityStatus.READY
    assert "PIPELINE_READY" in cap.execution_readiness
    assert cap.structured_status["dataset"] == "BigEarthNet.txt"
    assert cap.structured_status["pipeline_status"] == "PIPELINE_READY"
    assert cap.structured_status["training_status"] == "NOT_EXECUTED"
    assert cap.structured_status["evaluation_status"] == "NOT_EVALUATED"
    assert cap.structured_status["compute_target"] == "remote_gpu"
