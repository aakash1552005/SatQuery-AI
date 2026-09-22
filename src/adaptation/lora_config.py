"""
LoRA / PEFT Configuration for RS-VLM Adaptation on BigEarthNet.txt.
Implements parameter-efficient fine-tuning parameter schemas for GeoChat and multimodal remote sensing adapters.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any
import yaml


@dataclass
class RSLoraConfig:
    """Configuration for Low-Rank Adaptation (LoRA) on remote sensing vision-language models."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: Optional[list[str]] = field(default_factory=lambda: ["mm_projector"])

    # Training Hyperparameters
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True  # Preferred on modern NVIDIA Ada/Ampere/Hopper
    logging_steps: int = 10
    save_steps: int = 250
    seed: int = 42

    # Compute Target
    compute_target: str = "remote_gpu"  # "local_cpu", "local_gpu", "remote_gpu"
    model_name_or_path: str = "MBZUAI/geochat-7b"
    output_dir: str = "models/checkpoints/bigearthnet_txt_lora"

    def validate(self) -> list[str]:
        """Validate parameter ranges and requirements."""
        errors = []
        if self.r < 1:
            errors.append(f"LoRA rank r must be >= 1, got {self.r}")
        if self.lora_alpha < 1:
            errors.append(f"LoRA alpha must be >= 1, got {self.lora_alpha}")
        if not (0.0 <= self.lora_dropout < 1.0):
            errors.append(f"LoRA dropout must be in [0.0, 1.0), got {self.lora_dropout}")
        if not self.target_modules:
            errors.append("target_modules cannot be empty")
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {self.batch_size}")
        if self.gradient_accumulation_steps < 1:
            errors.append(f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard dictionary."""
        return asdict(self)

    def to_peft_config(self) -> Any:
        """Instantiate official peft.LoraConfig if peft is installed."""
        try:
            from peft import LoraConfig, TaskType
            task_type_enum = getattr(TaskType, self.task_type, TaskType.CAUSAL_LM)
            return LoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                bias=self.bias,
                task_type=task_type_enum,
                modules_to_save=self.modules_to_save,
            )
        except ImportError:
            return None


def load_lora_config(config_path: Path | str) -> RSLoraConfig:
    """Load and validate RSLoraConfig from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f)

    # Filter recognized fields
    recognized = {k: v for k, v in raw_dict.items() if hasattr(RSLoraConfig, k)}
    config = RSLoraConfig(**recognized)
    errs = config.validate()
    if errs:
        raise ValueError(f"Invalid LoRA configuration in {path}: {'; '.join(errs)}")
    return config
