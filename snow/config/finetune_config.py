
from dataclasses import dataclass

@dataclass
class DataConfig:
    dataset_path: tuple = None
    """Paths to the datasets."""

    modality_id: str = "YmBot"
    """Identity of robot which has corresponding modality in configuration."""

    processor_path: str = None
    """Path to the processor that preprocesses the language and vision data."""

    config_path: str = None
    """Path to the configuration file containing data, model, training."""


@dataclass
class ModelConfig:
    model_path: str = None
    """Path to the model which will be trained."""

    tune_llm: bool = True
    """Whether tune the language model."""

    tune_visual: bool = True
    """Whether tune the vision model."""



@dataclass
class TrainConfig:
    per_device_train_batch_size: int = 1
    """Batch size for single gpu training."""

    output_dir: str = None
    """Path to the output directory of the training."""

