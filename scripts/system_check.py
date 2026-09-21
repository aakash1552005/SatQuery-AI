"""
SatQuery AI — System Feasibility Check
=======================================
Section 3 / Section 43 of the v4 Master Specification.

Detects: OS, Python, PyTorch, CUDA, GPU, CPU, RAM, disk, Git, Docker.
Classifies machine into PROFILE A / B / C / D.
"""

import platform
import shutil
import subprocess
import sys
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str], timeout: int = 10) -> str | None:
    """Run a subprocess and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def get_os_info() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def get_python_info() -> dict:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def get_pytorch_info() -> dict:
    try:
        import torch

        info: dict = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        }

        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_vram_gb = round(props.total_mem / (1024**3), 2)
                try:
                    free_vram_bytes = torch.cuda.mem_get_info(i)[0]
                    free_vram_gb = round(free_vram_bytes / (1024**3), 2)
                except Exception:
                    free_vram_gb = None
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "total_vram_gb": total_vram_gb,
                    "free_vram_gb": free_vram_gb,
                    "compute_capability": f"{props.major}.{props.minor}",
                })
            info["gpus"] = gpus
        else:
            info["gpu_count"] = 0
            info["gpus"] = []

        return info

    except ImportError:
        return {"version": None, "error": "PyTorch not installed"}


def get_cpu_info() -> dict:
    return {
        "processor": platform.processor(),
        "physical_cores": os.cpu_count(),
    }


def get_ram_info() -> dict:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
            "percent_used": vm.percent,
        }
    except ImportError:
        # Fallback for Windows without psutil
        if platform.system() == "Windows":
            raw = _run(["wmic", "OS", "get", "TotalVisibleMemorySize", "/Value"])
            if raw:
                for line in raw.split("\n"):
                    if "TotalVisibleMemorySize" in line:
                        kb = int(line.split("=")[1].strip())
                        return {"total_gb": round(kb / (1024**2), 2), "available_gb": None, "percent_used": None}
        return {"total_gb": None, "error": "psutil not installed"}


def get_disk_info() -> dict:
    try:
        usage = shutil.disk_usage(Path.cwd())
        return {
            "path": str(Path.cwd()),
            "total_gb": round(usage.total / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "used_percent": round((usage.used / usage.total) * 100, 1),
        }
    except Exception as e:
        return {"error": str(e)}


def get_git_info() -> dict:
    version = _run(["git", "--version"])
    lfs = _run(["git", "lfs", "version"])
    return {
        "git_version": version,
        "git_lfs_version": lfs,
    }


def get_docker_info() -> dict:
    version = _run(["docker", "--version"])
    return {
        "docker_version": version,
        "available": version is not None,
    }


def classify_profile(pytorch_info: dict) -> str:
    """
    PROFILE A — ≥48GB VRAM class (A100/L20/A6000)
    PROFILE B — 12–24GB VRAM (high-end consumer GPU)
    PROFILE C — <12GB GPU or small-GPU dev machine
    PROFILE D — CPU only
    """
    if not pytorch_info.get("cuda_available", False):
        return "D"

    gpus = pytorch_info.get("gpus", [])
    if not gpus:
        return "D"

    max_vram = max(g.get("total_vram_gb", 0) for g in gpus)

    if max_vram >= 48:
        return "A"
    elif max_vram >= 12:
        return "B"
    else:
        return "C"


PROFILE_LABELS = {
    "A": "PROFILE A -- FULL GPU (>=48GB VRAM class -- A100/L20/A6000)",
    "B": "PROFILE B -- LIMITED GPU (12-24GB VRAM -- high-end consumer)",
    "C": "PROFILE C -- SMALL GPU (<12GB VRAM -- dev/laptop GPU)",
    "D": "PROFILE D -- CPU ONLY",
}


def determine_runtime_mode(profile: str) -> str:
    if profile == "A":
        return "FULL_AI"
    elif profile in ("B", "C"):
        return "HYBRID"
    else:
        return "DEMO_FALLBACK"


def print_report(report: dict) -> None:
    """Print the formatted feasibility report to stdout."""
    print("=" * 64)
    print("  SATQUERY AI -- SYSTEM FEASIBILITY REPORT")
    print("=" * 64)
    print(f"  Generated: {report['timestamp']}")
    print()

    # --- OS ---
    osi = report["os"]
    print(f"  Platform:          {osi['platform']}")
    print(f"  OS:                {osi['system']} {osi['release']}")
    print()

    # --- Python ---
    py = report["python"]
    print(f"  Python:            {py['version']} ({py['implementation']})")
    print(f"  Executable:        {py['executable']}")
    print()

    # --- PyTorch ---
    pt = report["pytorch"]
    if pt.get("version"):
        print(f"  PyTorch:           {pt['version']}")
        print(f"  CUDA available:    {pt.get('cuda_available', False)}")
        if pt.get("cuda_version"):
            print(f"  CUDA version:      {pt['cuda_version']}")
        if pt.get("cudnn_version"):
            print(f"  cuDNN version:     {pt['cudnn_version']}")
        print(f"  GPU count:         {pt.get('gpu_count', 0)}")
        for gpu in pt.get("gpus", []):
            free_str = f" (free: {gpu['free_vram_gb']}GB)" if gpu.get('free_vram_gb') is not None else ""
            print(f"    [{gpu['index']}] {gpu['name']} — {gpu['total_vram_gb']}GB VRAM{free_str} — CC {gpu['compute_capability']}")
    else:
        print(f"  PyTorch:           NOT INSTALLED")
        if pt.get("error"):
            print(f"    → {pt['error']}")
    print()

    # --- CPU / RAM / Disk ---
    cpu = report["cpu"]
    print(f"  CPU:               {cpu['processor']}")
    print(f"  Cores:             {cpu['physical_cores']}")

    ram = report["ram"]
    if ram.get("total_gb"):
        avail = f" (available: {ram['available_gb']}GB)" if ram.get("available_gb") else ""
        print(f"  RAM:               {ram['total_gb']}GB{avail}")
    else:
        print(f"  RAM:               unknown ({ram.get('error', '')})")

    disk = report["disk"]
    if disk.get("total_gb"):
        print(f"  Disk ({disk['path']}): {disk['free_gb']}GB free / {disk['total_gb']}GB total ({disk['used_percent']}% used)")
    print()

    # --- Git / Docker ---
    git = report["git"]
    print(f"  Git:               {git['git_version'] or 'NOT FOUND'}")
    print(f"  Git LFS:           {git['git_lfs_version'] or 'NOT FOUND'}")
    docker = report["docker"]
    print(f"  Docker:            {docker['docker_version'] or 'NOT AVAILABLE'}")
    print()

    # --- Profile ---
    profile = report["compute_profile"]
    print("-" * 64)
    print(f"  COMPUTE PROFILE:   {PROFILE_LABELS[profile]}")
    print(f"  RUNTIME MODE:      {report['runtime_mode']}")
    print("-" * 64)
    print()

    # --- Capability assessment ---
    print("  CAPABILITY READINESS (pre-model-preflight):")
    print()
    print(f"    SAR deterministic pathway:    READY (always -- no GPU)")
    print(f"    Optical-SAR fusion engine:    READY (always -- no GPU)")
    print(f"    Deterministic change engine:  READY (always -- no GPU)")
    print()

    if profile == "D":
        print(f"    GeoChat:                     UNAVAILABLE (no GPU)")
        print(f"                                  -> requires preflight on remote GPU")
    elif profile == "C":
        vram = 0
        if pt.get("gpus"):
            vram = pt["gpus"][0].get("total_vram_gb", 0)
        print(f"    GeoChat:                     REQUIRES PREFLIGHT")
        print(f"                                  -> {vram}GB VRAM may be insufficient")
        print(f"                                  -> 4-bit quantization needed, may still OOM")
        print(f"                                  -> remote GPU burst recommended")
    elif profile == "B":
        print(f"    GeoChat:                     REQUIRES PREFLIGHT")
        print(f"                                  -> likely feasible with 4-bit quantization")
    else:
        print(f"    GeoChat:                     REQUIRES PREFLIGHT")
        print(f"                                  -> likely feasible at fp16")

    print("    ChangeChat:                  BLOCKED_LICENSE (default)")
    print("                                  -> license gate must pass before preflight")
    print(f"    CROMA:                       DISABLED (optional, stretch only)")
    print()
    print(f"    BigEarthNet.txt:             NOT YET CHECKED")
    print(f"    RS adaptation status:        NOT_STARTED")
    print()

    if osi["system"] == "Windows":
        print("  [!] PLATFORM NOTE:")
        print("    Windows detected. WSL2 (Ubuntu 22.04+) is recommended for")
        print("    GIS/ML dependencies (GDAL, Rasterio, Flash Attention, AROSICS).")
        print("    Frontend/GUI runs on Windows host; backend/ML runs in WSL2.")
        print()

    print("=" * 64)
    print("  STATUS: FEASIBILITY REPORT COMPLETE")
    print("  NEXT:   Proceed to Day 1 implementation (Section 32)")
    print("=" * 64)


def main():
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "spec_version": "v4",
        "os": get_os_info(),
        "python": get_python_info(),
        "pytorch": get_pytorch_info(),
        "cpu": get_cpu_info(),
        "ram": get_ram_info(),
        "disk": get_disk_info(),
        "git": get_git_info(),
        "docker": get_docker_info(),
    }

    profile = classify_profile(report["pytorch"])
    report["compute_profile"] = profile
    report["runtime_mode"] = determine_runtime_mode(profile)

    # Print human-readable report
    print_report(report)

    # Save JSON for machine consumption
    out_path = Path(__file__).parent.parent / "data" / "system_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  JSON report saved to: {out_path}")


if __name__ == "__main__":
    main()
