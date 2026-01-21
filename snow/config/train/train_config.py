
from dataclasses import dataclass

@dataclass
class TrainConfig:
    per_device_train_batch_size: int = 1
    """Batch size for single gpu training."""

    output_dir: str = None
    """Path to the output directory of the training."""
