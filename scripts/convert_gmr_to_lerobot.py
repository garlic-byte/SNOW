import os
import shutil
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from dataclasses import dataclass
import numpy as np


@dataclass
class DatasetConfig:
    dataset_path: str = "/home/wsj/Desktop/data/dataset_origin/amass_retarget"
    """Path to the dataset"""

    repo_id: str = "amass/test"
    """A short but accurate description of the task performed during the recording"""

    fps: int = 20
    """Number of seconds for data recording for each episode."""

    root: str = "datasets/amass"
    """Root directory where the dataset will be stored"""

    robot_name: str = "YmBot"
    """Name of the dataset."""



def create_dataset(cfg: DatasetConfig) -> LeRobotDataset:
    # Create empty dataset or load existing saved episodes
    if os.path.exists(cfg.root):
        shutil.rmtree(cfg.root)

    action_features = {"action": {
            "dtype": "float32",
            "shape": (30,),
            "names": list(map(str, list(range(30)))),
        }
    }
    obs_features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    }
    dataset_features = {**action_features, **obs_features}
    dataset = LeRobotDataset.create(
        cfg.repo_id,
        cfg.fps,
        root=cfg.root,
        robot_type=cfg.robot_name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        batch_encoding_size=1,
    )
    return dataset

def record_loop(
    dataset: LeRobotDataset,
    single_task: str = None,
    imgs_data: np.ndarray = None,
    actions_data: np.ndarray = None,
):
    """Save one episode in the dataset."""
    num_episode = imgs_data.shape[0]
    # Check shape of action
    assert actions_data.shape[0] == num_episode, (
        f"actions_data.shape[0] should be equal to num_episode, but not {actions_data.shape[0]} != {num_episode}"
    )
    for episode_index in range(num_episode):
        observation_frame = {"observation.images.front": imgs_data[episode_index]}
        action_frame = {"action": actions_data[episode_index]}
        frame = {**observation_frame, **action_frame}
        dataset.add_frame(frame, task=single_task)
    dataset.save_episode()


def get_single_task_data(dataset_path):
    data_directory = os.path.join(dataset_path, "data")
    text_directory = os.path.join(dataset_path, "text")
    for file_name in os.listdir(data_directory):
        if not file_name.endswith(".npy"):
            continue
        # join path of file.npz and file.txt
        file_path = os.path.join(data_directory, file_name)
        file_name_prefix = os.path.splitext(file_name)[0]
        text_path = os.path.join(text_directory, file_name_prefix + '.txt')

        # Convert to lerobot data
        motion_data = np.load(file_path, allow_pickle=True)
        with open(text_path, "r") as f:
            content = f.readline()
        task = content.split('#')[0]
        imgs = motion_data['img']
        del motion_data['img']
        actions = np.concatenate([motion for motion in motion_data.values()], axis=1, dtype=np.float32)
        yield task, imgs, actions


def main():
    # Ready for dataset
    cfg = DatasetConfig()
    lerobot_dataset = create_dataset(cfg)

    # Start to save lerobot dataset
    [
        record_loop(
            dataset=lerobot_dataset,
            single_task=task,
            imgs_data=imgs,
            actions_data = actions,
        ) for task, imgs, actions in get_single_task_data(cfg.dataset_path)
    ]

    lerobot_dataset.clear_episode_buffer()


if __name__ == "__main__":
    main()