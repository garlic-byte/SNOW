import os
import transformers
from snow.experient import run_train
from snow.config.finetune_config import (
    ModelConfig,
    DataConfig,
    TrainConfig,
)


def main():
    parser = transformers.HfArgumentParser(
        (ModelConfig, DataConfig, TrainConfig,)
    )
    model_config, data_config, train_config = parser.parse_args_into_dataclasses()

    # Modify configuration for single gpu
    data_config.dataset_path = ("/home/wsj/Desktop/code/VLA/robot/datasets/libero_10", )
    data_config.modality_id = "libero_panda"

    model_config.model_path = "/home/wsj/Downloads/weights/qwen3-vl-2b"
    model_config.tune_llm = False
    model_config.tune_visual = False
    model_config.test_mode = True

    # Static configuration
    project_name = "libero"
    data_config.processor_path = model_config.model_path
    model_config.lora_alpha = model_config.lora_rank * 2
    train_config.global_batch_size = train_config.per_device_train_batch_size * train_config.num_gpus
    train_config.output_dir = f"./outputs/{project_name}/gpus_{train_config.num_gpus}_batch_size_{train_config.global_batch_size * train_config.gradient_accumulation_steps}_mask_ratio_{data_config.mask_ratio}"

    # Initialize global rank parameters
    data_config.config_output_dir = os.path.join(train_config.output_dir, "configs")


    # Start training
    run_train(model_config, data_config, train_config)


if __name__ == "__main__":
    main()