
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


from snow.config import ROBOT_CONFIG
from snow.utils import process_parquet_files_optimized, get_frames_by_indices


# DIRECTORY
DATA_DIR = "data"
META_DIR = "meta"
VIDEO_DIR = "video"

# META INFO FILES
EPISODES_FILENAME = "episodes.jsonl"
EPISODES_STATS_FILENAME = "episodes_stats.jsonl"
INFO_FILENAME = "info.json"
TASKS_FILENAME = "tasks.jsonl"
MODALITY_FILENAME = "modality.json"
STATS_FILENAME = "stats.json"

# Convert index to language
ORIGINAL_KEY = "task_index"
ORIGINAL_TASK = "task"

# DATA INFO FILES
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


class LerobotDataset:
    def __init__(
            self,
            dataset_path: str,
            modality_id: str,
            video_backend: str = "ffmpeg",
            video_backend_kwargs: dict = None,
        ):
        self.dataset_path = Path(dataset_path)
        self.robot_modality = ROBOT_CONFIG[modality_id]
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.load_meta_info()


    def load_meta_info(self):
        """Load all meta information from the path of dataset."""
        META_PATH = self.dataset_path / META_DIR
        with open(META_PATH / EPISODES_FILENAME, 'r') as f:
            self.episodes_meta = [json.loads(line) for line in f]

        with open(META_PATH / EPISODES_STATS_FILENAME, 'r') as f:
            self.episodes_stats_meta = [json.loads(line) for line in f]

        with open(META_PATH / INFO_FILENAME, 'r') as f:
            self.info_meta = json.load(f)

        with open(META_PATH / TASKS_FILENAME, 'r') as f:
            self.tasks_meta = [json.loads(line) for line in f]
            self.task_map = {task_info[ORIGINAL_KEY]: task_info[ORIGINAL_TASK] for task_info in self.tasks_meta}

        with open(META_PATH / MODALITY_FILENAME, 'r') as f:
            self.modality_meta = json.load(f)

        if not (META_PATH / STATS_FILENAME).exists():
            process_parquet_files_optimized(self.dataset_path, STATS_FILENAME)
        with open(META_PATH / STATS_FILENAME, 'r') as f:
            self.stats_meta = json.load(f)

        # Initialize dependency parameters
        self.chunks_size = self.info_meta["chunks_size"]
        self.features = self.info_meta["features"]



    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episodes_meta)

    def _load_parquet_data(self, episode_index) -> pd.DataFrame:
        """Load single parquet in the dataset."""
        episode_chunk = episode_index // self.chunks_size
        parquet_path = self.dataset_path / DATA_PATH.format(episode_chunk=episode_chunk, episode_index=episode_index)
        origin_df = pd.read_parquet(parquet_path)
        action_language_df = pd.DataFrame()

        # Add language into data
        assert "language" in self.robot_modality, f"Language should be supplied in Robot modality."
        action_language_df["annotation.language"] = origin_df[ORIGINAL_KEY].apply(lambda x: self.task_map[x])

        # Save partly actions from modality
        assert "action" in self.robot_modality and "action" in self.modality_meta, f"Action should be supplied in Robot config and modality."
        for modality_key, modality_position in self.modality_meta["action"].items():
            action_language_df[f"action.{modality_key}"] = origin_df["action"].map(lambda x: x[modality_position["start"]:modality_position["end"]])
        return action_language_df

    def _load_images_data(self, episode_index: int, video_indices: np.ndarray) -> pd.DataFrame:
        """Load images data in the dataset."""
        video_data = pd.DataFrame()
        episode_chunk = episode_index // self.chunks_size
        for video_key in self.modality_meta["video"]:
            original_key = self.modality_meta["video"][video_key]["original_key"]
            assert original_key in self.features, f"Video {video_key} is not in lerobot dataset."
            video_path = self.dataset_path / VIDEO_PATH.format(episode_chunk=episode_chunk, video_key=original_key, episode_index=episode_index)
            frames_arr = get_frames_by_indices(
                video_path,
                video_indices,
                video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs,
            )
            video_data[f"video.{video_key}"] = list(frames_arr)
        return video_data


    def __getitem__(self, item) -> pd.DataFrame:
        """
        Get data of an episode from the dataset.
        :return dataframe contain:
            actions: shape of (length_episode, length_action),
            language: shape of (length_episode, 1),
            images: shape of (length_episode, height_video, width_video, 3),
        """
        assert item < len(self.episodes_meta), f"get lerobot episode index {item} > length of dataset."
        episode_meta = self.episodes_meta[item]
        episode_index = episode_meta["episode_index"]
        length = episode_meta["length"]

        # Load action from parquet
        episode_data = self._load_parquet_data(episode_index)

        # Load images from videos
        effective_length = min(length, len(episode_data))
        video_indices = np.arange(effective_length)
        video_data = self._load_images_data(episode_index, video_indices)
        episode_data = pd.concat([episode_data, video_data], axis=1)

        return episode_data



if __name__ == "__main__":
    dataset_path = "/home/wsj/Desktop/code/VLA/SNOW/datasets/amass"
    modality_id = "YmBot"
    video_backend = "ffmpeg"
    dataset = LerobotDataset(dataset_path=dataset_path, modality_id=modality_id, video_backend=video_backend)
    data = dataset[0]
    print(data)


