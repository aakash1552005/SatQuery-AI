#!/usr/bin/env bash
# SatQuery AI -- Remote GPU Training Launch Script
set -e

echo "=== SATQUERY AI: REMOTE GPU LORA TRAINING ==="
nvidia-smi

export SATQUERY_DATA_ROOT="${SATQUERY_DATA_ROOT:-./data}"
echo "Using SATQUERY_DATA_ROOT=${SATQUERY_DATA_ROOT}"

pip install -r requirements.txt

python -m torch.distributed.run --nproc_per_node=1 \
    scripts/run_day4_full.py \
    --train \
    --seed 42
