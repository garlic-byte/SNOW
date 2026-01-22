
from snow.config import SnowConfig, DataConfig, TrainConfig
from snow.data.data_pipeline import DataPipeline


def run_train(model_config: SnowConfig, data_config: DataConfig, train_config: TrainConfig):
    # Step 1 Create datasets, processor, collator
    data_pipeline = DataPipeline(data_config)


    # Step 2. load pre-trained model

    # Step 3. ready train-args