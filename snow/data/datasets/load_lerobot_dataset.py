
import json
import math
import os
import time
from pathlib import Path
import random
from typing import Dict

from numpy.random import Generator

import numpy as np
import pandas as pd
from tests.test_generic_params import Schedule

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
        dataset_index: int,
        dataset_path: str,
        modality_id: str,
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict = None,
        vessel_length: int = 10,
    ):
        self.dataset_index = dataset_index
        self.dataset_path = Path(dataset_path)
        self.robot_modality = ROBOT_CONFIG[modality_id]
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.load_meta_info()

        # Split step indices into (split_episode_nums) shard_vessels from each episode
        self.shard_vessel = None
        self.vessel_length = vessel_length


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
        self.total_episodes = self.info_meta["total_episodes"]
        self.chunks_size = self.info_meta["chunks_size"]
        self.features = self.info_meta["features"]
        self.transform = None
        self.action_dimension = self.robot_modality["action"].delta_indices


    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return self.total_episodes

    def get_stats(self):
        """
        Return the statistics of the dataset.
        :return dictory mapping action-joint-group -> stats-key -> stats-value:
            {
                action.root_pos: {
                    'max': np.ndarray, shape of (length_joint_group)
                }
            }
        """
        stats = {}

        assert 'action' in self.stats_meta, "stats_meta does not contain 'action' key."
        for modality_key in self.robot_modality['action'].modality_keys:
            start_index = self.modality_meta['action'][modality_key]['start']
            end_index = self.modality_meta['action'][modality_key]['end']
            stats[f"{modality_key}"] = {}
            for stats_key in ['max', 'min']:
                stats[f"{modality_key}"][stats_key] = (
                    np.array(self.stats_meta['action'][stats_key][start_index:end_index], dtype=np.float32)
                )

        return stats

    def get_modality_info(self):
        """Get the modality information."""
        return self.modality_meta['action']

    def get_dataset_index(self):
        """Get the dataset index."""
        return self.dataset_index

    def _load_parquet_data(self, episode_index) -> pd.DataFrame:
        """Load single parquet in the dataset."""
        episode_chunk = episode_index // self.chunks_size
        parquet_path = self.dataset_path / DATA_PATH.format(episode_chunk=episode_chunk, episode_index=episode_index)
        origin_df = pd.read_parquet(parquet_path)
        action_language_df = pd.DataFrame()

        # Check and add language into data
        assert "language" in self.robot_modality, f"Language should be supplied in Robot modality."
        assert len(self.robot_modality["language"].modality_keys) == 1 and len(self.robot_modality["language"].delta_indices) == 1, (
            f"Length of language modality_keys and delta_indices should be equal to 1"
        )
        language_modality_key = self.robot_modality["language"].modality_keys[0]
        action_language_df[f"language.{language_modality_key}"] = origin_df[ORIGINAL_KEY].apply(lambda x: self.task_map[x])

        # Save partly actions from modality
        assert "action" in self.robot_modality and "action" in self.modality_meta, f"Action should be supplied in Robot config and modality."
        for modality_key, modality_position in self.modality_meta["action"].items():
            action_language_df[f"action.{modality_key}"] = origin_df["action"].map(lambda x: x[modality_position["start"]:modality_position["end"]])
        return action_language_df

    def _load_images_data(
        self,
        episode_index: int,
        video_indices: np.ndarray | list[int]
    ) -> dict[str, np.ndarray]:
        """Load images data in the dataset."""
        video_data = {}
        episode_chunk = episode_index // self.chunks_size
        for video_key in self.modality_meta["video"]:
            original_key = self.modality_meta["video"][video_key]["original_key"]
            assert original_key in self.features, f"Video {video_key} is not in lerobot dataset."
            video_path = self.dataset_path / VIDEO_PATH.format(
                episode_chunk=episode_chunk, video_key=original_key, episode_index=episode_index
            )
            # t1 = time.time()
            frames_arr = get_frames_by_indices(
                video_path,
                video_indices,
                video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs,
            )
            # print(f"extract {len(frames_arr)} frames time: {time.time() - t1}")
            video_data[f"{video_key}"] = frames_arr.squeeze()
        return video_data


    def _get_episode_data(self, episode_index) -> pd.DataFrame:
        """
        Get data of an episode from the dataset.
        :return dataframe contain:
            actions: shape of (length_episode, length_action),
            language: shape of (length_episode, 1),
            images: shape of (length_episode, height_video, width_video, 3),
        """
        assert episode_index < len(self.episodes_meta), f"get lerobot episode index {episode_index} > length of dataset."
        episode_meta = self.episodes_meta[episode_index]
        episode_index = episode_meta["episode_index"]

        # Load action, state, language from parquet
        return self._load_parquet_data(episode_index)

    def get_episode_effect_length(self, episode_index: int) -> int:
        """Get effective episode length"""
        return max(0, self.episodes_meta[episode_index]["length"] - len(self.action_dimension) + 1)

    def count_total_frames(self) -> int:
        """Count the total number of frames in the dataset."""
        return sum([self.get_episode_effect_length(episode_index) for episode_index in range(self.total_episodes)])


    def initialize_shard_vessel(self, rng: Generator):
        """Initialize shard contains (dataset_index, episode_index, step_indices)."""
        shard_vessel = []
        for episode_index in range(self.total_episodes):
            # Generate effect indices
            step_effect_length = self.get_episode_effect_length(episode_index)
            step_indices = np.arange(0, step_effect_length)
            rng.shuffle(step_indices)

            # Split indices into predetermined length
            step_nums = math.ceil(step_effect_length / self.vessel_length)
            for step_index in range(step_nums):
                shard_vessel.append((self.dataset_index, episode_index, step_indices[step_index::step_nums]))

        return shard_vessel

    def get_step_data(self, episode_data: pd.DataFrame, episode_index: int, step_index: int):
        """
        Get single step data from the dataset.
        :param episode_data: data for one parquet episode contains language, action, state
        :param episode_index: index of episode
        :param step_index: index of step
        :return: step_data
            {
                'language':
                    {
                        'task': str,
                    }
                'action':
                    {
                        'root_pos': np.ndarray, shape of (action_horizon, 3),
                        'root_rot': np.ndarray, shape of (action_horizon, 4),
                        'dof_pos': np.ndarray, shape of (action_horizon, 23),
                    }
                'observation.images':
                    {
                        'front': np.ndarray, shape of (height_image, width_image, 3),
                    }
            }
        """
        # Extract data of single step
        step_data = {}
        for modality_type in self.robot_modality:
            # extracting images only when we need them
            if modality_type == "observation.images":
                step_data[modality_type] = self._load_images_data(episode_index, [step_index])
                continue

            step_data[modality_type] = {}
            for modality_key in self.robot_modality[modality_type].modality_keys:
                # indices of needing extract modality data, special for action which has horizon
                delta_indices = self.robot_modality[modality_type].delta_indices

                # Extract steps from dataframe
                step_delta_indices = [step_index + delta_index for delta_index in delta_indices]
                modality_data = episode_data[f"{modality_type}.{modality_key}"].iloc[step_delta_indices]

                # Convert series to np.ndarray
                step_data[modality_type][modality_key] = np.vstack(
                    [modality_data.iloc[i] for i in delta_indices], dtype=np.float32
                ) if modality_type in ["state", "action"] else modality_data.iloc[0]

        return step_data

    def get_steps_data(self, episode_index: int, step_indices: np.array) -> list:
        """
        Get multiple steps data of an episode from the dataset.
        For language, action and state, we extract all steps data into pd,
        but for images, we only extract needed step frame into numpy arrays.
        This trick will speed up data loading.
        """
        episode_parquet_data = self._get_episode_data(episode_index)

        return [
            self.get_step_data(episode_parquet_data, episode_index, step_index)
            for step_index in step_indices
        ]


if __name__ == "__main__":

    dataset = LerobotDataset(
        dataset_index=0,
        dataset_path="/home/wsj/Desktop/code/VLA/SNOW/datasets/amass",
        modality_id="YmBot",
        video_backend="torchcodec"
    )

    count_time = []
    for i in range(1000):
        start_time = time.time()
        data = dataset.get_steps_data(0, np.arange(0, 10))
        end_time = time.time()
        print(f"step {i} took {end_time - start_time:2f} seconds.")
        count_time.append(end_time - start_time)
    print(f"average time per episode: {sum(count_time) / len(count_time):2f} seconds.")



