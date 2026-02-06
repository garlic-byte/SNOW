from dataclasses import dataclass

@dataclass
class DataConfig:
    """Data configuration contains data and process."""

    dataset_path: str = None
    """Paths to the datasets."""

    modality_id: str = "YmBot"
    """Identity of robot which has corresponding modality in configuration."""

    processor_path: str = None
    """Path to the processor that preprocesses the language and vision data."""

    config_path: str = None
    """Path to the configuration file containing data, model, training."""

    video_backend: str = "torchcodec"
    """Video backend to use."""

    video_backend_kwargs: dict = None
    """Keyword arguments passed to ffmpeg."""

    shard_size: int = 2**10
    """Shard size to use."""

    vessel_length: int = 10
    """Length of each vessel."""

    seed: int = 64
    """Seed for random number generator."""

    inter_size: tuple[int, int] = (256, 256)
    """Size of interim image."""

    crop_fraction: float = 0.95
    """Fraction of images to crop."""

    target_size: tuple[int, int] = (224, 224)
    """Size of target image."""

    color_jitter: bool = True
    """Whether to apply color jitter."""

    config_output_dir: str = None
    """Path to the output directory for configuration."""
