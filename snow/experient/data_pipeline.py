
from snow.config import SnowConfig, DataConfig, TrainConfig

class DataPipeline:
    """
    Data pipeline for training and evaluation contains:
        1. load multiple datasets
        2. design processor for images, language and action
        3. use collator stack modalities.
    """
    def __init__(self, data_config: DataConfig):
        pass