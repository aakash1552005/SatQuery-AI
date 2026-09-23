# Day 4 ML Evaluation Report

## Hardware
- **Host Platform**: Windows 10 (10.0.26200-SP0), AMD64 Architecture
- **CPU**: AMD64 Family 25 Model 124 Stepping 0 (12 logical cores)
- **System RAM**: 15.27 GB Total (Available: ~4.00 GB)
- **Local GPU**: None (`torch.cuda.is_available() == False`, `torch.cuda.device_count() == 0`)
- **GPU Name**: AMD Radeon 740M Graphics (Integrated APU, 512 MB VRAM, no CUDA support)
- **VRAM**: 0.00 GB CUDA VRAM
- **CUDA**: NOT AVAILABLE
- **PyTorch**: 2.13.0+cpu
- **Transformers**: 5.14.1
- **PEFT**: 0.21.0
- **bitsandbytes**: NOT INSTALLED (CUDA required)
- **Accelerate**: 1.15.0
- **Disk Free Space**: 98.72 GB on Drive C: (No external D:, E:, or F: drives detected)
- **SATQUERY_DATA_ROOT**: Local repository path (`./data`)
- **VRAM Constraint Note**: VRAM requirement is configuration-dependent. The exact requirement must be measured for the selected model, quantization, image resolution, context length, batch size, gradient accumulation, and checkpointing configuration.

## Dataset
- **Name**: BigEarthNet.txt (arXiv:2603.29630)
- **Role**: Primary Training / Fine-tuning Dataset
- **Version**: Official BigEarthNet v2.0 Release 1.0 Parquet (`BigEarthNet.txt.parquet`, 466,819,745 bytes)
- **Total Parquet Records**: 9,553,962 records across 464,044 unique Sentinel-1/Sentinel-2 patch pairs
- **License**: CDLA-Permissive-1.0
- **Data Reality Status**: 4 local development patch pairs physically present and verified; 1,000 / 300 / 300 benchmark manifests are currently METADATA_ONLY locally pending external high-capacity drive mount.

## Training Samples
- **Manifest**: `data/manifests/ben_train_subset.json`
- **Total Records**: 1,000
- **Usable S1**: 0 locally on Drive C: (1,000 metadata-only)
- **Usable S2**: 0 locally on Drive C: (1,000 metadata-only)
- **Usable Paired Samples**: 0 locally (Requires external high-capacity mount via `SATQUERY_DATA_ROOT`)
- **Task Balance**: 250 binary, 250 MCQ, 250 bounding box, 250 captioning

## Validation Samples
- **Manifest**: `data/manifests/ben_val_subset.json`
- **Total Records**: 300
- **Usable S1**: 0 locally on Drive C: (300 metadata-only)
- **Usable S2**: 0 locally on Drive C: (300 metadata-only)
- **Usable Paired Samples**: 0 locally (Requires external high-capacity mount via `SATQUERY_DATA_ROOT`)
- **Task Balance**: 75 binary, 75 MCQ, 75 bounding box, 75 captioning

## Test Samples
- **Manifest**: `data/manifests/ben_test_subset.json`
- **Total Records**: 300
- **Usable S1**: 0 locally on Drive C: (300 metadata-only)
- **Usable S2**: 0 locally on Drive C: (300 metadata-only)
- **Usable Paired Samples**: 0 locally (Untouched, strictly isolated test split)
- **Cross-Split Leakage**: 0 overlapping samples or patch pairs across train/val/test

## Model
- **Target Model**: SatQuery-RS-VLM-LoRA
- **Base Architecture**: RS-VLM (`MBZUAI/geochat-7b`, LLaVA-1.5 multimodal remote sensing foundation architecture)
- **Weights Present**: False (GeoChat 7B weights not hosted on local disk or in HF cache)
- **Model Load Test**: FAIL (Offline execution; 7B base weights cannot be loaded on Profile D CPU host)
- **Processor Load Test**: NOT_EXECUTED
- **Inference Test**: NOT_EXECUTED

## LoRA Configuration
- **Rank ($r$)**: 16
- **Alpha ($\alpha$)**: 32
- **Dropout**: 0.05
- **Target Modules**: `["q_proj", "v_proj", "k_proj", "o_proj"]`
- **Bias**: `none`
- **Task Type**: `CAUSAL_LM`
- **Modules to Save**: `["mm_projector"]`
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$)
- **Learning Rate**: $2 \times 10^{-4}$
- **Weight Decay**: 0.01
- **Warmup Ratio**: 0.03
- **Scheduler**: Cosine with warmup
- **Precision**: bf16 (GPU target) / fp32 (CPU fallback)
- **Batch Size**: 2 per device
- **Gradient Accumulation**: 4
- **Epochs**: 3
- **Seed**: 42

## Training Time
- **Value**: 0.0 seconds
- **Status**: BLOCKED — Local CPU-only infrastructure without CUDA accelerator

## Steps
- **Value**: 0 / 1,000 steps executed locally
- **Status**: BLOCKED — Local CPU-only host

## Best Checkpoint
- **Value**: NONE
- **Status**: BLOCKED — No checkpoint physically exists on local disk; zero fake checkpoints fabricated

## Validation Loss
- **Value**: N/A — NOT EXECUTED
- **Status**: BLOCKED — Parameter training not executed on local host

## Accuracy
- **Value**: N/A — NOT EXECUTED

## Balanced Accuracy
- **Value**: N/A — NOT EXECUTED

## Precision
- **Value**: N/A — NOT EXECUTED

## Recall
- **Value**: N/A — NOT EXECUTED

## Specificity
- **Value**: N/A — NOT EXECUTED

## Macro-F1
- **Value**: N/A — NOT EXECUTED

## Weighted-F1
- **Value**: N/A — NOT EXECUTED

## Grounding IoU
- **Mean IoU**: N/A — NOT EXECUTED
- **Median IoU**: N/A — NOT EXECUTED

## Acc@0.5
- **Value**: N/A — NOT EXECUTED

## Acc@0.7
- **Value**: N/A — NOT EXECUTED

## BLEU-4
- **Value**: N/A — NOT EXECUTED

## METEOR
- **Value**: N/A — NOT EXECUTED

## ROUGE-L
- **Value**: N/A — NOT EXECUTED

## CIDEr
- **Value**: N/A — NOT EXECUTED

## VQA Score
- **Overall VQA Score**: N/A — NOT EXECUTED
- **Binary Question Score**: N/A — NOT EXECUTED
- **MCQ Question Score**: N/A — NOT EXECUTED
- **Per-Sensor Breakdown (SAR vs Optical)**: N/A — NOT EXECUTED

## Seed 42
- **Validation Loss**: N/A — NOT EXECUTED
- **Macro-F1**: N/A — NOT EXECUTED
- **Mean IoU**: N/A — NOT EXECUTED
- **BLEU-4**: N/A — NOT EXECUTED
- **VQA Score**: N/A — NOT EXECUTED

## Seed 123
- **Validation Loss**: N/A — NOT EXECUTED
- **Macro-F1**: N/A — NOT EXECUTED
- **Mean IoU**: N/A — NOT EXECUTED
- **BLEU-4**: N/A — NOT EXECUTED
- **VQA Score**: N/A — NOT EXECUTED

## Seed 2026
- **Validation Loss**: N/A — NOT EXECUTED
- **Macro-F1**: N/A — NOT EXECUTED
- **Mean IoU**: N/A — NOT EXECUTED
- **BLEU-4**: N/A — NOT EXECUTED
- **VQA Score**: N/A — NOT EXECUTED

## Mean
- **Repeated-Seed Validation Mean**: N/A — NOT EXECUTED

## Std
- **Repeated-Seed Validation Std**: N/A — NOT EXECUTED

## Baseline
- **Model**: MBZUAI/geochat-7b (Pretrained base RS-VLM without domain adaptation)
- **Status**: NOT_EXECUTED (Cannot execute 7B inference locally without CUDA GPU and >16 GB RAM)
- **Baseline Metrics**: N/A — NOT EXECUTED

## Adapted Model
- **Model**: SatQuery-RS-VLM-LoRA (PEFT r=16 on BigEarthNet.txt)
- **Status**: PENDING_REMOTE_GPU_EXECUTION
- **Adapted Metrics**: N/A — NOT EXECUTED

## Failure Analysis
Because model training has not been executed on this CPU host, empirical model error analysis cannot be evaluated from model outputs. However, input validation checks and physical raster audits across the 1,600 sampled benchmark records and engineering fixtures revealed the following structural failure boundaries:
1. **Missing Local Rasters**: The 1,600 benchmark records in `ben_train_subset.json`, `ben_val_subset.json`, and `ben_test_subset.json` are metadata-only on Drive C: due to disk space constraints (>350 GB raw imagery); attempting to open missing rasters triggers strict preflight rejection.
2. **Dual-Pol Ratio Formulation**: SAR cross-polarization ratio requires linear power domain calculation ($10^{(\text{VV}_{\text{dB}} - \text{VH}_{\text{dB}})/10}$); direct decibel division $\text{VV}_{\text{dB}}/\text{VH}_{\text{dB}}$ is physically invalid and blocked by mathematical contracts.
3. **Radiometric Quantization**: Raw optical imagery (Sentinel-2 16-bit surface reflectance) must be preserved as reflectance $[0.0, 1.0]$; naive 8-bit truncation causes severe radiometric loss.
4. **Coordinate Normalization**: Grounding tokens are formatted in normalized range $[0, 100]$; bounding box parses are validated against image dimensions $[H, W]$ to guard against out-of-bounds coordinates.
5. **Modal Sufficiency**: Temporal change queries or optical-SAR fusion queries presented with a single timestamp or single modality trigger honest refusal responses.

---

## Final Acceptance Status
DAY4_SOFTWARE_COMPLETE_TRAINING_BLOCKED
