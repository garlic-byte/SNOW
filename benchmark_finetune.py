import os
import transformers
from snow.experient import run_train
from snow.config import SnowConfig, DataConfig, TrainConfig
from snow.utils import initialize_dist, save_dataclass


def main():
    parser = transformers.HfArgumentParser(
        (SnowConfig, DataConfig, TrainConfig,)
    )
    model_config, data_config, train_config = parser.parse_args_into_dataclasses()

    # Modify configuration for single gpu
    data_config.dataset_path = "/home/wsj/Desktop/data/dataset/rough_datasets/accad_xxx"
    data_config.modality_id = "YmBot"
    data_config.processor_path = "/home/wsj/Downloads/weights/qwen3-vl-2b" # "/home/wsj/Downloads/weights/qwen25-vl-3b"
    data_config.max_action_dim = model_config.max_action_dim
    data_config.max_action_horizon = model_config.action_horizon
    model_config.model_name = "qwen3-vl"
    model_config.model_path = "weights"
    model_config.tune_llm = False
    model_config.tune_visual = False
    model_config.tune_top_llm_layers = 0
    # model_config.tune_projector = False
    # model_config.tune_vlln = False

    train_config.learning_rate = 1e-5
    train_config.weight_decay = 1e-5
    train_config.warmup_ratio = 0.05
    train_config.max_grad_norm = 1.0
    train_config.per_device_train_batch_size = 2
    train_config.gradient_accumulation_steps = 1



    # Static configuration
    project_name = "x-0214-test"
    train_config.global_batch_size = train_config.per_device_train_batch_size * train_config.num_gpus
    train_config.output_dir = (f"./outputs/{project_name}/gpus_{train_config.num_gpus}_batch_size_"
                               f"{train_config.global_batch_size * train_config.gradient_accumulation_steps}")

    # Initialize global rank parameters
    data_config.config_output_dir = os.path.join(train_config.output_dir, "configs")
    initialize_dist()

    # Save all configurations
    save_dataclass(
        data_config.config_output_dir,
        model_config=model_config,
        data_config=data_config,
        train_config=train_config,
    )

    # Start training
    run_train(model_config, data_config, train_config)


if __name__ == "__main__":
    main()