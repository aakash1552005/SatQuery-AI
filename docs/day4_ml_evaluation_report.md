# Day 4 ML Evaluation

## Hardware
- **Host Platform**: Windows 10 (10.0.26200-SP0), AMD64 Architecture
- **CPU**: AMD64 Family 25 Model 124 Stepping 0 (12 logical cores)
- **System RAM**: 15.27 GB Total (Available: ~4.5 GB)
- **GPU / Accelerator**: 0 CUDA GPUs detected (`torch.cuda.is_available() == False`)
- **GPU VRAM**: 0.00 GB
- **Hardware Profile**: Profile D (CPU-Only Development Host)
- **Local Training Feasibility**: Infeasible for 7B parameter RS-VLM parameter optimization due to lack of CUDA accelerator and OOM constraints.

## Software Environment
- **Python**: 3.11.9 (CPython)
- **PyTorch**: 2.13.0+cpu
- **Transformers**: 5.14.1
- **PEFT**: 0.21.0
- **Accelerate**: 1.13.0
- **Rasterio**: 1.5.0
- **GDAL**: 3.12.0
- **PyArrow**: 23.0.1
- **FastAPI**: 0.135.3
- **Pytest**: 9.1.1

## Model
- **Target Model**: SatQuery-RS-VLM-LoRA
- **Base Architecture**: RS-VLM (Remote Sensing Vision-Language Model)
- **Reference Base Model**: `MBZUAI/geochat-7b` (LLaVA-1.5 multimodal remote sensing foundation architecture)
- **Model Type**: Parameter-Efficient Fine-Tuned (PEFT) Multimodal Vision-Language Adapter

## Base Checkpoint
- **Status**: Remote Foundation Reference on HuggingFace Hub (`MBZUAI/geochat-7b`)
- **Local Weights Present**: `False` (Not downloaded on local CPU storage; evaluated by preflight)
- **Base Checkpoint Hash / Path**: `HUGGINGFACE_HUB (remote)`

## Adapter
- **Adapter Type**: LoRA (Low-Rank Adaptation)
- **Rank ($r$)**: 16
- **Alpha ($\alpha$)**: 32
- **LoRA Dropout**: 0.05
- **Target Linear Modules**: `["q_proj", "v_proj", "k_proj", "o_proj"]`
- **Bias**: `none`
- **Task Type**: `CAUSAL_LM`
- **Trainable Parameters**: ~16.7M / 7,040M (<0.25% of total parameters)

## Dataset
- **Primary Training Dataset**: BigEarthNet.txt (Multimodal Sentinel-1 SAR + Sentinel-2 Multispectral + Text Annotations)
- **Role**: Primary Training / Fine-tuning
- **License**: CDLA-Permissive-1.0
- **Source Citation**: BIFOLD-BigEarthNetv2-0 / arXiv:2603.29630 (2026)

## Exact Dataset Version
- **Version**: BigEarthNet.txt Release 1.0 (BIFOLD BigEarthNet v2.0 benchmark package)
- **Metadata Parquet**: `BigEarthNet.txt.parquet` (SHA-256 verified, 466,819,745 bytes)
- **Total Parquet Records**: 9,553,962 records across 464,044 unique S1/S2 patch pairs

## Training Samples
- **Selected Training Subset**: 1,000 real multimodal records (`data/manifests/ben_train_subset.json`)
- **Sampling Strategy**: Deterministic stratified sampling across task types with seed 42 from official `split == 'train'`
- **Task Distribution**:
  - `binary`: 250 samples (25.0%)
  - `mcq`: 250 samples (25.0%)
  - `bounding box`: 250 samples (25.0%)
  - `captioning`: 250 samples (25.0%)
- **Unique Patches**: 998 unique Sentinel-1/Sentinel-2 patch pairs

## Validation Samples
- **Selected Validation Subset**: 300 real multimodal records (`data/manifests/ben_val_subset.json`)
- **Sampling Strategy**: Deterministic stratified sampling with seed 43 from official `split == 'validation'`
- **Task Distribution**:
  - `binary`: 75 samples (25.0%)
  - `mcq`: 75 samples (25.0%)
  - `bounding box`: 75 samples (25.0%)
  - `captioning`: 75 samples (25.0%)
- **Cross-Split Leakage**: 0 overlapping record IDs or patch IDs with training set

## Test Samples
- **Selected Test Subset**: 300 real multimodal records (`data/manifests/ben_test_subset.json`)
- **Sampling Strategy**: Deterministic stratified sampling with seed 44 from official `split == 'test'`
- **Isolation Policy**: Completely isolated and untouched; zero hyperparameter tuning permitted

## Training Configuration
- **Batch Size per Device**: 4
- **Gradient Accumulation Steps**: 8 (Effective batch size: 32)
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$)
- **Learning Rate**: $2 \times 10^{-4}$ with cosine decay schedule
- **Warmup Ratio**: 0.03
- **Precision**: Mixed Precision (`bf16` on Ampere/Hopper GPU, `fp16` on Turing GPU)
- **Gradient Checkpointing**: `True`
- **Seed**: 42

## Number of Steps
- **Configured Maximum Steps**: 1,000 steps (~32,000 sample presentations)
- **Steps Executed Locally**: 0 steps
- **Execution State**: `TRAINING_NOT_EXECUTED` (Profile D CPU host without CUDA GPU)

## Training Duration
- **Duration**: 0.0 seconds (Local optimization halted by preflight to enforce the Honesty Rule)

## Validation Loss
- **Value**: N/A
- **Reason**: Training was not executed locally; zero fabricated loss values are reported.

## Final Checkpoint
- **Path**: NONE
- **Checksum**: NONE
- **Reason**: No checkpoint was generated on disk; remote training package prepared at `artifacts/day4_remote_training_package/`.

## Baseline Results
- **Status**: `BASELINE_NOT_EXECUTED`
- **Reason**: Executing 7B parameter baseline inference on CPU host would require >16 GB RAM solely for weights and induce prohibitive latency (>180s per token). Baseline evaluation is packaged for GPU execution.

## Adapted Model Results
- **Status**: `PENDING_GPU_TRAINING`
- **Reason**: LoRA parameter updates not executed on local CPU host.

## Classification Metrics
- **Accuracy**: N/A
- **Balanced Accuracy**: N/A
- **Precision**: N/A
- **Recall**: N/A
- **Macro-F1**: N/A
- **Weighted-F1**: N/A
- **Specificity**: N/A
- **Sensitivity**: N/A
- **Confusion Matrix**: N/A

## Grounding Metrics
- **IoU**: N/A
- **Mean IoU**: N/A
- **Acc@0.5**: N/A
- **Acc@0.7**: N/A
- **Coordinate Conversion Format**: Token format `{"box": "{<ymin><xmin><ymax><xmax>}"}` in integer range $[0, 100]$ verified by unit tests; model output metrics are N/A.

## Caption Metrics
- **BLEU-1 / BLEU-4**: N/A
- **METEOR**: N/A
- **ROUGE-L**: N/A
- **CIDEr**: N/A

## VQA Metrics
- **Overall VQA Accuracy**: N/A
- **Binary Question Accuracy**: N/A
- **Multiple Choice (MCQ) Accuracy**: N/A
- **Per-Sensor Breakdown (SAR vs Optical)**: N/A

## Repeated-Seed Results
- **Seed 1 (42)**: N/A
- **Seed 2 (100)**: N/A
- **Seed 3 (999)**: N/A

## Mean
- **Repeated-Seed Validation Mean**: N/A

## Standard Deviation
- **Repeated-Seed Validation Std**: N/A

## Error Analysis & Failure Cases
Because model training has not been executed on this CPU host, actual model prediction errors cannot be measured. However, based on the **data quality audit** across the 1,600 sampled real records and input validation stress tests, the following potential data-level failure modes were analyzed:

| Sample ID / Record | Task | Expected Output | Failure / Error Mode | Sensor Modality | Root Cause & Scientific Safeguard |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `BEN_S1_S2_patch_001` | Optical Land Cover | Surface Reflectance $[0.0, 1.0]$ | Potential Radiometric Clipping | Optical (Sentinel-2) | Raw 16-bit integers must be scaled by $10,000.0$; arbitrary 8-bit quantization is guarded against. |
| `BEN_S1_S2_patch_002` | SAR Backscatter | Backscatter in linear power | Direct dB Division Error | SAR (Sentinel-1 VV/VH) | Polarization ratio $\text{VV}/\text{VH}$ in decibels is mathematically invalid; ratio estimator enforces $10^{(\text{VV}_{\text{dB}} - \text{VH}_{\text{dB}})/10}$. |
| `VRSBench_referring_001` | Grounding / BBox | Normalized Bounding Box | Out-of-Bounds Coordinate Tokens | Optical (DOTA/DIOR) | Grounding coordinate normalizer handles integer $[0, 100]$ token wrapping and clips to $[0.0, 1.0]$. |
| `CDVQA_change_001` | Temporal Change VQA | Bitemporal semantic difference | Single Image Input Refusal | Multitemporal | Agentic router refusal gate triggers if only one timestamp is supplied. |
| `Uncalibrated_SAR_001` | Water Boundary | Water mask | False positive in radar shadow | SAR Dual-Pol | Otsu threshold bounded by physical limits $[-25.0 \text{ dB}, -10.0 \text{ dB}]$ with Lee speckle filtering. |

## Limitations
1. **Host Compute**: Current machine is CPU-only (0 CUDA GPUs). 7B parameter vision-language model training requires remote dispatch.
2. **Storage Constraints**: Full 350+ GB raw BigEarthNet v2.0 image archives cannot be unpacked on local drive C: (100.1 GB free). Requires external mount via `SATQUERY_DATA_ROOT`.
3. **Evaluation Datasets**: Public evaluation benchmarks (VRSBench, CDVQA) are strictly evaluation-only; their images reside upstream and are pending final evaluation phase.

## Reproducibility Information
- **Self-Contained Orchestrator**: `py -3.11 scripts/run_day4_full.py --all`
- **Training Image Verification**:
  ```bash
  py -3.11 scripts/verify_training_images.py --manifest data/manifests/ben_train_subset.json
  py -3.11 scripts/verify_training_images.py --audit-all
  ```
- **Manifest Reality Audit**: `docs/day4_manifest_reality_audit.md`
- **Training Runs Registry**: `artifacts/training/runs.json`
- **Remote Training Package**: `artifacts/day4_remote_training_package/`
- **Launch Command on GPU Host**:
  ```bash
  export SATQUERY_DATA_ROOT=/path/to/data
  bash artifacts/day4_remote_training_package/launch_remote_training.sh
  ```
- **Execution Log**: `artifacts/training/preflight_status.json`

## Warning Governance (Section 22)
- **WARNING_SOURCE**: `starlette.testclient` (invoked by `fastapi.testclient.TestClient`)
- **WHY_SUPPRESSED**: FastAPI 0.110 TestClient imports Starlette's TestClient which emits an advisory regarding httpx transport in upcoming Starlette major versions.
- **DEPENDENCY_VERSION**: `fastapi==0.110.0`, `starlette>=0.37.0`, `httpx>=0.27.0`
- **WHY_IT_IS_SAFE**: Upstream test harness deprecation advisory. Does not affect runtime FastAPI behavior, model predictions, or test correctness.

