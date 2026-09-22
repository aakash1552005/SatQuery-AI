"""
Training Preflight Checker for RS-VLM LoRA Fine-Tuning.
Verifies hardware, accelerator availability, VRAM capacity, and software prerequisites.
Follows the Honesty Rule: never fabricates training feasibility on CPU-only hosts.
"""

from dataclasses import dataclass, field
import platform
import psutil
from typing import Optional, Any
from src.adaptation.lora_config import RSLoraConfig

try:
    import torch
except ImportError:
    torch = None

try:
    import transformers
except ImportError:
    transformers = None

try:
    import peft
except ImportError:
    peft = None

try:
    import accelerate
except ImportError:
    accelerate = None


@dataclass
class TrainingPreflightReport:
    """Detailed hardware and software feasibility assessment for training."""
    host_os: str
    python_version: str
    cpu_count: int
    system_ram_gb: float
    cuda_available: bool
    gpu_count: int
    gpu_name: Optional[str]
    total_gpu_vram_gb: Optional[float]
    torch_version: Optional[str]
    transformers_version: Optional[str]
    peft_version: Optional[str]
    accelerate_version: Optional[str]

    training_feasible: bool
    status: str  # "TRAINING_READY", "PIPELINE_READY_REMOTE_GPU", "BLOCKED_DEPENDENCY"
    reasons: list[str] = field(default_factory=list)
    recommended_action: str = ""


def check_training_feasibility(config: Optional[RSLoraConfig] = None) -> TrainingPreflightReport:
    """
    Perform rigorous training preflight audit.
    Determines whether local hardware supports RS-VLM LoRA training or whether the
    pipeline should be marked PIPELINE_READY_REMOTE_GPU.
    """
    host_os = f"{platform.system()} {platform.release()}"
    py_ver = platform.python_version()
    cpu_count = psutil.cpu_count(logical=True) or 1
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    cuda_available = False
    gpu_count = 0
    gpu_name = None
    gpu_vram_gb = None
    torch_ver = torch.__version__ if torch else None
    trans_ver = transformers.__version__ if transformers else None
    peft_ver = peft.__version__ if peft else None
    accel_ver = accelerate.__version__ if accelerate else None

    if torch and torch.cuda.is_available():
        cuda_available = True
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_vram_gb = round(vram_bytes / (1024 ** 3), 2)

    reasons = []
    training_feasible = False
    status = "PIPELINE_READY_REMOTE_GPU"
    recommended_action = ""

    # Assess dependencies
    if not torch or not transformers or not peft:
        missing = []
        if not torch: missing.append("torch")
        if not transformers: missing.append("transformers")
        if not peft: missing.append("peft")
        status = "BLOCKED_DEPENDENCY"
        reasons.append(f"Missing required training dependencies: {', '.join(missing)}")
        recommended_action = "Install missing packages before running pipeline."
        return TrainingPreflightReport(
            host_os=host_os,
            python_version=py_ver,
            cpu_count=cpu_count,
            system_ram_gb=ram_gb,
            cuda_available=cuda_available,
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            total_gpu_vram_gb=gpu_vram_gb,
            torch_version=torch_ver,
            transformers_version=trans_ver,
            peft_version=peft_ver,
            accelerate_version=accel_ver,
            training_feasible=False,
            status=status,
            reasons=reasons,
            recommended_action=recommended_action,
        )

    # Assess compute feasibility
    if not cuda_available or (gpu_vram_gb is not None and gpu_vram_gb < 14.0):
        training_feasible = False
        status = "PIPELINE_READY_REMOTE_GPU"
        if not cuda_available:
            reasons.append("Current host Profile D is CPU-only (0 CUDA GPUs).")
            reasons.append("Fine-tuning 7B parameter RS-VLM models with multi-modal inputs requires CUDA acceleration.")
            recommended_action = "Dispatch training to remote NVIDIA GPU (Profile A/B with >=16GB VRAM) using bigearthnet_txt_lora.yaml."
        else:
            reasons.append(f"Detected GPU ({gpu_name}) has {gpu_vram_gb} GB VRAM, which is below the 16 GB threshold for RS-VLM adaptation.")
            recommended_action = "Enable QLoRA 4-bit quantization or migrate to remote GPU with >=16 GB VRAM."
    else:
        training_feasible = True
        status = "TRAINING_READY"
        reasons.append(f"Host has adequate CUDA acceleration ({gpu_name}, {gpu_vram_gb} GB VRAM).")
        recommended_action = "Execute LoRA training using RSVLMAdaptationPipeline."

    return TrainingPreflightReport(
        host_os=host_os,
        python_version=py_ver,
        cpu_count=cpu_count,
        system_ram_gb=ram_gb,
        cuda_available=cuda_available,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        total_gpu_vram_gb=gpu_vram_gb,
        torch_version=torch_ver,
        transformers_version=trans_ver,
        peft_version=peft_ver,
        accelerate_version=accel_ver,
        training_feasible=training_feasible,
        status=status,
        reasons=reasons,
        recommended_action=recommended_action,
    )
