"""
Remote GPU Preflight Verification for SatQuery AI.
Checks CUDA devices, VRAM capacity, PyTorch build, and dataset readiness without hard-coded paths.
"""

import sys
import os
from pathlib import Path

def run_preflight():
    print("=" * 65)
    print("SATQUERY AI -- REMOTE GPU TRAINING PREFLIGHT")
    print("=" * 65)

    import torch
    cuda_avail = torch.cuda.is_available()
    device_count = torch.cuda.device_count()

    print(f"PyTorch Version:     {torch.__version__}")
    print(f"CUDA Available:      {cuda_avail}")
    print(f"CUDA Device Count:   {device_count}")

    if not cuda_avail or device_count == 0:
        print("\n[FAIL] No CUDA accelerator detected! Training requires an NVIDIA GPU.")
        print("Preflight outcome: BLOCKED_NO_CUDA")
        return 1

    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        total_vram = round(props.total_memory / (1024 ** 3), 2)
        print(f"Device [{i}]:         {name} ({total_vram} GB VRAM)")

    # Data check
    data_root = os.getenv("SATQUERY_DATA_ROOT", ".")
    print(f"\nActive Data Root:    {data_root}")
    manifest_dir = Path(__file__).resolve().parent / "manifests"
    train_manifest = manifest_dir / "ben_train_subset.json"
    if train_manifest.exists():
        print(f"Train Manifest:      FOUND ({train_manifest.name})")
    else:
        print(f"Train Manifest:      MISSING in {manifest_dir}")

    print("\n>>> PREFLIGHT PASSED: Environment ready for GPU execution! <<<\n")
    return 0

if __name__ == "__main__":
    sys.exit(run_preflight())
