
from snow.config import DataConfig
from snow.data import ShardCacheDataset
from snow.data.collator.collator import DataCollator
from snow.data.transformer.transformer import Transformer


class DataPipeline:
    """
    Data pipeline for training and evaluation contains:
        1. load multiple datasets
        2. design processor for images, language and action
        3. use collator stack modalities.
    """
    def __init__(self, data_config: DataConfig):
        # Load dataset
        self.dataset = ShardCacheDataset(
            dataset_paths=data_config.dataset_path,
            modality_id=data_config.modality_id,
            video_backend=data_config.video_backend,
            video_backend_kwargs=data_config.video_backend_kwargs,
            shard_size=data_config.shard_size,
            seed=data_config.seed,
            vessel_length=data_config.vessel_length,
            config_output_dir=data_config.config_output_dir,
        )
        # Design processor
        self.transformer = Transformer(
            processor_path=data_config.processor_path,
            inter_size=data_config.inter_size,
            crop_fraction=data_config.crop_fraction,
            target_size=data_config.target_size,
            color_jitter=data_config.color_jitter,
            modality_id=data_config.modality_id,
            statistics=self.dataset.get_statistics(),
            max_action_dim=data_config.max_action_dim,
        )
        self.transformer.train()
        self.dataset.set_transform(self.transformer)

        # Use collator
        self.collator = DataCollator(
            processor_path=data_config.processor_path,
        )

    def eval(self):
        self.transformer.eval()

