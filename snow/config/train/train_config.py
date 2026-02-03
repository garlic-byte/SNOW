
from dataclasses import dataclass

@dataclass
class TrainConfig:
    per_device_train_batch_size: int = 2
    """Batch size for single gpu training."""

    num_gpus: int = 1
    """Number of gpus used for training."""

    gradient_accumulation_steps: int = 4
    """Number of steps used for gradient accumulation."""

    output_dir: str = None
    """Path to the output directory of the training."""

    max_steps: int = int(3e4)
    """Maximum number of training steps."""

    learning_rate: float = 1e-4
    """Learning rate for optimizer."""

    dataloader_num_workers: int = 1
    """Number of dataloader workers."""

    deepspeed_config: str = None
    """DeepSpeed config for training."""
