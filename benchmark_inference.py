import os.path
import time
from dataclasses import dataclass
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from snow.config import SnowConfig, DataConfig, TrainConfig
from snow.data import LerobotDataset
from snow.data.collator.collator import DataCollator
from snow.data.data_pipeline import DataPipeline
from snow.model.snow_model import SnowModel
from snow.utils import read_configs
from snow.data.transformer.transformer import Transformer


@dataclass
class ConfigInference:
    data_path: str = "/home/wsj/Desktop/code/VLA/robot/datasets/libero_10"
    model_path: str = "outputs/x-0214-libero-drift/gpus_1_batch_size_200"
    modality_keys: tuple = ("arm",)
    save_plot_dir: str = "outputs/x-0214-libero-drift/plot"


def plot_trajectory(
    modality_info: dict,
    save_plot_path: str = None,
):
    if save_plot_path is not None:
        matplotlib.use("Agg")

    gt_action_across_time = modality_info["gt_action_across_time"]
    pred_action_across_time = modality_info["pred_action_across_time"]
    modality_keys = modality_info["modality_keys"]
    trajectory_index = modality_info["trajectory_index"]
    mse = modality_info["mse"]
    action_horizon = modality_info["action_horizon"]

    step, action_dim = gt_action_across_time.shape
    # Adjust figure size and spacing to accommodate titles
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2))

    # Leave plenty of space at the top for titles
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    print("Creating visualization...")

    # Combine all modality keys into a single string
    # add new line if total length is more than 60 chars
    modality_string = ""
    for key in modality_keys:
        modality_string += key + "\n " if len(modality_string) > 40 else key + ", "
    title_text = f"Trajectory Analysis - ID: {trajectory_index}\nModalities: {modality_string[:-2]}\nUnnormalized MSE: {mse:.6f}"

    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#2E86AB", y=0.95)

    # Loop through each action dim
    for i, ax in enumerate(axes):
        # The dimensions of state_joints and action are the same only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        ax.plot(gt_action_across_time[:, i], label="gt action", linewidth=2)
        ax.plot(pred_action_across_time[:, i], label="pred action", linewidth=2)

        # put a dot every ACTION_HORIZON
        for j in range(0, step, action_horizon):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, i], "ro", label="inference point", markersize=6)
            else:
                ax.plot(j, gt_action_across_time[j, i], "ro", markersize=4)

        ax.set_title(f"Action Dimension {i}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Set better axis labels
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)

    if save_plot_path:
        print("saving plot to", save_plot_path)
        if not os.path.exists(os.path.dirname(save_plot_path)):
            os.makedirs(os.path.dirname(save_plot_path))
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def calc_mse_for_single_trajectory(
    dataset: LerobotDataset,
    transform: Transformer,
    collator: DataCollator,
    trajectory_index: int,
    policy: nn.Module,
    modality_keys: tuple,
    plot: bool = True,
    save_plot_dir: str = None,
):
    gt_action_across_time = []
    pred_action_across_time = []
    single_trajectory_len = dataset.get_episode_effect_length(trajectory_index)
    single_trajectory = dataset.get_steps_data(trajectory_index, np.arange(0, single_trajectory_len))
    action_horizon = policy.config.action_horizon
    action_modality = dataset.get_modality_info()
    # In each single timesteps, concatenate complete action values of each modality keys
    cur_index = 0
    inference_time = []
    while cur_index < single_trajectory_len:
        step_data = collator(
            [transform(single_trajectory[cur_index])]
        ).to(policy.device)

        start_time = time.perf_counter()
        predict_action = policy.get_action(step_data).action_pred
        end_time = time.perf_counter()
        inference_time.append(end_time - start_time)
        print("predict action time:", end_time - start_time, 's')

        target_action = single_trajectory[cur_index]['action']
        predict_action = transform.decode_action(predict_action)
        # Action has horizon according modality
        for i in range(action_horizon):
            gt_action = []
            pred_action = []
            for key in modality_keys:
                assert (target_action[key][i].shape == predict_action[key][i].shape), \
                    (f"Get gt shape is not equal to pred shape: "
                     f"{target_action[key][i].shape} != {predict_action[key][i].shape}")

                gt_action.append(target_action[key][i])
                pred_action.append(predict_action[key][i])

            gt_action_across_time.append(np.concatenate(gt_action, axis=0))
            pred_action_across_time.append(np.concatenate(pred_action, axis=0))
        cur_index += action_horizon

    print("inference avg time:", (sum(inference_time) / len(inference_time)), 's')
    gt_action_across_time = np.array(gt_action_across_time)
    pred_action_across_time = np.array(pred_action_across_time)

    # calc MSE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)


    if plot:
        info = {
            'gt_action_across_time': gt_action_across_time,
            'pred_action_across_time': pred_action_across_time,
            "modality_keys": modality_keys,
            "trajectory_index": trajectory_index,
            'mse': mse,
            "action_horizon": action_horizon
        }
        plot_trajectory(info, os.path.join(save_plot_dir, f"{trajectory_index}.png"))
    return mse

def inference(config: ConfigInference):
    configs_name = ['data_config', 'model_config', 'train_config', 'stats']
    configs = {}
    for config_name in configs_name:
        configs[config_name] = read_configs(config.model_path, config_name)
    data_config = DataConfig(**configs['data_config'])

    dataset = LerobotDataset(
                dataset_index=0,
                dataset_path=config.data_path,
                modality_id=data_config.modality_id,
            )
    transformer = Transformer(
        processor_path=data_config.processor_path,
        inter_size=data_config.inter_size,
        crop_fraction=data_config.crop_fraction,
        target_size=data_config.target_size,
        color_jitter=data_config.color_jitter,
        modality_id=data_config.modality_id,
        statistics=configs['stats'],
        max_action_dim=configs['model_config']["max_action_dim"],
        max_action_horizon=configs['model_config']["action_horizon"],
    )
    transformer.eval()
    collator = DataCollator(
        processor_path=data_config.processor_path,
    )

    model_config = configs['model_config']
    model_config["tune_llm"] = False
    model_config["tune_visual"] = False
    model_config["tune_top_llm_layers"] = 0
    model_config["tune_projector"] = False
    model_config["tune_diffusion_model"] = False
    model_config["tune_vlln"] = False
    model_config = SnowConfig(**model_config)
    model = SnowModel.from_pretrained(config.model_path, config=model_config).to(device='cuda')
    model.set_eval()
    for i in range(10):
        calc_mse_for_single_trajectory(
            dataset=dataset,
            transform=transformer,
            collator=collator,
            trajectory_index=i,
            policy=model,
            modality_keys=config.modality_keys,
            plot=True,
            save_plot_dir=config.save_plot_dir,
        )


if __name__ == "__main__":
    cfg = ConfigInference()
    inference(cfg)