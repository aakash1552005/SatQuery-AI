# SatQuery AI -- Checkpoint Management & Deployment Instructions

## Overview
This directory describes the checkpoint structure, persistence strategy, resumption mechanisms, and deployment workflow for the SatQuery AI LoRA adapter trained on BigEarthNet.txt.

## Checkpoint Layout
During training via `train.py`, checkpoints are saved in `--output-dir <DIR>` with the following structure:

```
checkpoints/run_001/
├── checkpoint-step-250/
│   ├── adapter_config.json        # LoRA configuration (r=16, alpha=32, target modules)
│   ├── adapter_model.safetensors  # Trainable LoRA weights
│   ├── training_args.bin          # PyTorch / Trainer execution state
│   ├── optimizer.pt               # Optimizer state (for resumability)
│   ├── scheduler.pt               # Learning rate scheduler state
│   └── trainer_state.json         # Step counter, epoch, loss history
├── best_checkpoint/               # Symlink or copy of lowest validation loss checkpoint
└── final_checkpoint/              # Checkpoint saved at the conclusion of all epochs
```

## Resuming Training
To resume an interrupted training run from the latest checkpoint:
```bash
python train.py \
    --config configs/bigearthnet_txt_lora.yaml \
    --train-manifest manifests/ben_train_subset.json \
    --val-manifest manifests/ben_val_subset.json \
    --output-dir checkpoints/run_001 \
    --resume-from-checkpoint checkpoints/run_001/checkpoint-step-250
```

## Validating & Testing a Checkpoint
To run evaluation against the untouched test split:
```bash
python evaluate.py \
    --checkpoint checkpoints/run_001/best_checkpoint \
    --manifest manifests/ben_test_subset.json \
    --output-dir evaluation_results
```

Expected evaluation outputs:
- `evaluation_results/test_metrics.json`: Accuracy, Balanced Accuracy, Precision, Recall, Macro-F1, Weighted-F1, Grounding IoU, BLEU, METEOR, ROUGE-L, CIDEr, VQA score.
- `evaluation_results/confusion_matrix.json`: Multiclass / multilabel confusion matrix.
- `evaluation_results/error_analysis.json`: Detailed error breakdown across failure modes.

## Deploying Back to Local SatQuery AI Host
Once trained on the remote GPU host:
1. Copy the contents of `best_checkpoint/` (`adapter_config.json` and `adapter_model.safetensors`) into the local repository directory:
   ```
   models/checkpoints/bigearthnet_txt_lora/
   ```
2. Copy `evaluation_results/test_metrics.json` into:
   ```
   artifacts/evaluation/runs.json
   ```
3. Update `docs/day4_ml_evaluation_report.md` with observed metrics and change acceptance status from `DAY4_SOFTWARE_COMPLETE_TRAINING_BLOCKED` to `DAY4_FULLY_COMPLETE`.
