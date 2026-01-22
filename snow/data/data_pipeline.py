
from snow.config import SnowConfig, DataConfig, TrainConfig
from snow.data import ShardCacheDataset


class DataPipeline:
    """
    Data pipeline for training and evaluation contains:
        1. load multiple datasets
        2. design processor for images, language and action
        3. use collator stack modalities.
    """
    def __init__(self, data_config: DataConfig):
        # Load dataset
        dataset = ShardCacheDataset(
            dataset_paths=data_config.dataset_path,
            modality_id=data_config.modality_id,
            video_backend=data_config.video_backend,
            video_backend_kwargs=data_config.video_backend_kwargs,
            shard_size=data_config.shard_size,
            seed=data_config.seed,
        )
        # Design processor

