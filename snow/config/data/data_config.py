from dataclasses import dataclass

@dataclass
class DataConfig:
    """Data configuration contains data and process."""

    dataset_path: tuple = None
    """Paths to the datasets."""

    modality_id: str = "YmBot"
    """Identity of robot which has corresponding modality in configuration."""

    processor_path: str = None
    """Path to the processor that preprocesses the language and vision data."""

    config_path: str = None
    """Path to the configuration file containing data, model, training."""

    num_gpus: int = 1
    """Number of GPUs to use."""

    video_backend: str = "ffmpeg"
    """Video backend to use."""

    video_backend_kwargs: dict = None
    """Keyword arguments passed to ffmpeg."""