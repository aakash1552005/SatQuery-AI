# SatQuery AI -- Remote GPU Training & Evaluation Package

## Overview
This directory contains the self-contained, reproducible remote GPU execution package for fine-tuning and evaluating **SatQuery-RS-VLM-LoRA** on multimodal remote sensing data (**BigEarthNet.txt**, arXiv:2603.29630) for SIH Problem Statement 26167 (ISRO / SAC).

When local host infrastructure lacks a CUDA GPU or sufficient VRAM, this package provides a 1-to-1 portable runtime that can be copied to any cloud VM, HPC node, or workstation equipped with an NVIDIA GPU.

## Package Contents
```
artifacts/day4_remote_training_package/
├── README.md                      # Comprehensive execution & deployment guide
├── requirements.txt               # Pinned Python package dependencies
├── environment.yml                # Conda environment definition with PyTorch CUDA
├── preflight.py                   # Preflight script: tests GPU, CUDA, RAM, VRAM & data
├── train.py                       # LoRA fine-tuning script with resume & checkpointing
├── evaluate.py                    # Multi-task evaluation script for test/val splits
├── metrics.py                     # Metric calculations (F1, IoU, BLEU, ROUGE, CIDEr, VQA)
├── run.sh                         # Linux/Cloud automated launcher (preflight -> train -> eval)
├── run.ps1                        # Windows PowerShell automated launcher
├── configs/
│   └── bigearthnet_txt_lora.yaml  # Complete LoRA r=16 training hyperparameters
├── manifests/
│   ├── ben_train_subset.json      # Stratified training split (1,000 samples)
│   ├── ben_val_subset.json        # Stratified validation split (300 samples)
│   └── ben_test_subset.json       # Untouched test split (300 samples)
└── checkpoint_instructions/
    └── README.md                  # Checkpoint inspection, resumption, and handoff instructions
```

## System Requirements
- **Compute**: NVIDIA GPU with CUDA support (e.g. RTX 3090 / 4090, A10G, A100, H100).
- **VRAM**: VRAM requirement is configuration-dependent. The exact requirement must be measured for the selected model, quantization, image resolution, context length, batch size, gradient accumulation, and checkpointing configuration. (16GB+ recommended for fp16/bf16 with batch size 2 and gradient accumulation 4).
- **Disk Space**: At least 30 GB free space for base model weights, dataset imagery, and checkpoints.
- **Python**: Python 3.10 or 3.11 with `torch >= 2.1.0` and CUDA toolkit.

## Setup Instructions

### Option 1: Conda Environment
```bash
conda env create -f environment.yml
conda activate satquery-gpu
```

### Option 2: Pip Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset Configuration
Set the `SATQUERY_DATA_ROOT` environment variable to point to the root directory containing the BigEarthNet GeoTIFF imagery:
```bash
# Linux / macOS
export SATQUERY_DATA_ROOT=/path/to/data

# Windows PowerShell
$env:SATQUERY_DATA_ROOT = "D:\data"
```
Ensure the folder structure contains:
```
$SATQUERY_DATA_ROOT/
├── BigEarthNet-S1/
└── BigEarthNet-S2/
```

## Execution

### Step 1: Preflight Verification
Verify GPU availability, compute capabilities, and manifest integrity:
```bash
python preflight.py
```

### Step 2: Automated End-to-End Run
Run preflight, training, and evaluation in sequence:
- **On Linux / Cloud**:
  ```bash
  bash run.sh
  ```
- **On Windows**:
  ```powershell
  .\run.ps1
  ```

### Step 3: Manual Step-by-Step Training & Evaluation
```bash
# 1. Train LoRA adapter
python train.py \
    --config configs/bigearthnet_txt_lora.yaml \
    --train-manifest manifests/ben_train_subset.json \
    --val-manifest manifests/ben_val_subset.json \
    --output-dir checkpoints/run_001 \
    --seed 42

# 2. Evaluate on untouched test split
python evaluate.py \
    --checkpoint checkpoints/run_001/best_checkpoint \
    --manifest manifests/ben_test_subset.json \
    --output-dir evaluation_results
```

## Generated Outputs
- `checkpoints/run_001/best_checkpoint/`: Trained LoRA adapter (`adapter_config.json`, `adapter_model.safetensors`).
- `evaluation_results/test_metrics.json`: Evaluated metrics across all tasks.
- `evaluation_results/confusion_matrix.json`: Detailed classification confusion matrix.
- `evaluation_results/error_analysis.json`: Analysis of failed predictions.
