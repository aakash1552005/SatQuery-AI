import sys
import shutil

print("=" * 60)
print("SATQUERY AI -- GPU PREFLIGHT & HARDWARE DISCOVERY")
print("=" * 60)

try:
    import torch
    cuda_avail = torch.cuda.is_available()
    dev_count = torch.cuda.device_count()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "NO GPU"
    print(f"PyTorch Version:     {torch.__version__}")
    print(f"CUDA Available:      {cuda_avail}")
    print(f"Device Count:        {dev_count}")
    print(f"Device Name:         {dev_name}")
except ImportError as e:
    print(f"PyTorch: NOT INSTALLED ({e})")
    cuda_avail = False
    dev_count = 0
    dev_name = "NO GPU"

for pkg in ["transformers", "peft", "bitsandbytes", "accelerate"]:
    try:
        mod = __import__(pkg)
        print(f"{pkg.capitalize() + ' Version:':20} {getattr(mod, '__version__', 'Installed')}")
    except ImportError:
        print(f"{pkg.capitalize() + ' Version:':20} NOT INSTALLED")

try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"System RAM:          {vm.total / (1024**3):.2f} GB (Available: {vm.available / (1024**3):.2f} GB)")
except ImportError:
    print("System RAM:          psutil not installed")

usage = shutil.disk_usage(".")
print(f"Disk Free Space:     {usage.free / (1024**3):.2f} GB (Total: {usage.total / (1024**3):.2f} GB)")
print("=" * 60)
