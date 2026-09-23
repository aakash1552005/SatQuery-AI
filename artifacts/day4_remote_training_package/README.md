# SatQuery AI -- Remote GPU Training Package

This directory contains the self-contained production package for fine-tuning
SatQuery-RS-VLM-LoRA on an NVIDIA GPU cluster (e.g. A100 / H100 / RTX 4090 with >= 16 GB VRAM).

## Files Included:
- `bigearthnet_txt_lora.yaml`: LoRA rank 16 configuration
- `requirements.txt`: Exact dependency pins
- `manifests/`: Deterministic BigEarthNet.txt stratified subsets
- `launch_remote_training.sh`: Single-command GPU training launcher

## Execution:
```bash
bash launch_remote_training.sh
```
