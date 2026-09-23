#!/usr/bin/env bash
# SatQuery AI -- Remote GPU Training and Evaluation Launcher (Linux / Cloud)
set -e

echo "=========================================================="
echo "SatQuery AI -- Remote GPU Training & Evaluation Launcher"
echo "=========================================================="

export SATQUERY_DATA_ROOT="${SATQUERY_DATA_ROOT:-$(pwd)/data}"
echo "Using SATQUERY_DATA_ROOT: $SATQUERY_DATA_ROOT"

# 1. Preflight
python preflight.py

# 2. Train LoRA Adapter
python train.py \
    --config configs/bigearthnet_txt_lora.yaml \
    --train-manifest manifests/ben_train_subset.json \
    --val-manifest manifests/ben_val_subset.json \
    --output-dir checkpoints/run_001 \
    --seed 42

# 3. Evaluate Best Checkpoint
python evaluate.py \
    --checkpoint checkpoints/run_001/best_checkpoint \
    --manifest manifests/ben_test_subset.json \
    --output-dir evaluation_results

echo "Execution completed successfully!"
