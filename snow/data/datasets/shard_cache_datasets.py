import time
from collections import defaultdict

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
import torch.distributed as dist
from concurrent.futures import Future, ThreadPoolExecutor

from snow.data.datasets import LerobotDataset
from snow.utils import save_dataclass


class ShardCacheDataset(IterableDataset):
    def __init__(
        self,
        dataset_paths: str,
        modality_id: str = None,
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict = None,
        shard_size: int = 2**10,
        seed: int = 64,
        vessel_length: int = 10,
        config_output_dir: str = None,
    ):
        super().__init__()
        self.dataset_paths = dataset_paths.split(',')
        self.shard_size = shard_size
        self.seed = seed
        self.config_output_dir = config_output_dir

        # Initialize all datasets
        self.datasets = [LerobotDataset(
                dataset_index=dataset_index,
                dataset_path=dataset_path,
                modality_id=modality_id,
                video_backend=video_backend,
                video_backend_kwargs=video_backend_kwargs,
                vessel_length=vessel_length,
            )for dataset_index, dataset_path in enumerate(self.dataset_paths)]
        self.epoch = -1

        # Initialize world size and current rank
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.worker_id = None
        self.num_workers = None

        # Initialize parameters of multiple threads
        self._cache_job = None
        self._executor = None
        self.cur_schedule_index = 0
        self.filter_schedules = []
        self.transformer = None

    def get_statistics(self) -> dict[str, np.ndarray]:
        """Get and merge statistics which only contain 'max' and 'min' values of action."""

        modality_meta_action = self.datasets[0].get_modality_info()
        # Collect all stats
        total_stats = {}
        for dataset in self.datasets:
            # Varify modalities are same
            assert modality_meta_action == dataset.get_modality_info(), (
                f"dataset {dataset.get_dataset_index()} modality information is not same."
            )
            stats = dataset.get_stats()
            for modality_key in stats:
                # Initialize total stats
                if modality_key not in total_stats:
                    total_stats[modality_key] = defaultdict(list)
                # Collect all information into list
                for stats_key, stats_value in stats[modality_key].items():
                    total_stats[modality_key][stats_key].append(stats_value)

        # Calculate all stats
        final_stats = {}
        for modality_key in total_stats:
            final_stats[modality_key] = {}
            final_stats[modality_key]['max'] = np.max(np.vstack(total_stats[modality_key]['max']), axis=0)
            final_stats[modality_key]['min'] = np.min(np.vstack(total_stats[modality_key]['min']), axis=0)
        save_dataclass(self.config_output_dir, stats=final_stats)
        return final_stats


    def set_transform(self, transformer):
        """API for setting the transform which can process images, languages and actions."""
        self.transformer = transformer

    def _reset_schedules(self):
        """Create random schedules for loading datasets."""
        self.epoch += 1
        self.cur_schedule_index = 0
        rng = np.random.default_rng(self.seed + self.epoch)

        # Initialize schedules (dataset_index, episode_index, step_indices)
        random_schedules = []
        for datas in self.datasets:
            random_schedules += datas.initialize_shard_vessel(rng)

        rng.shuffle(random_schedules, axis=0)

        # Initialize current identity worker and number of workers
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        if self.worker_id is None:
            self.worker_id = worker_id
            self.num_workers = num_workers
        else:
            assert self.worker_id == worker_id and self.num_workers == num_workers, (
                f"worker_id or num_workers has changed, expected {self.worker_id=} {self.num_workers=} "
                f"but {worker_id=} {num_workers=}."
            )

        # Filter schedules for multiple ranks and dataloaders
        filter_schedules = []
        for index in range(len(random_schedules)):
            if index % (self.world_size * self.num_workers) == self.rank * self.num_workers + self.worker_id:
                filter_schedules.append(random_schedules[index])
        print(self.rank, self.worker_id, len(filter_schedules))
        self.filter_schedules = filter_schedules


    def _shard_vessel(self):
        """Load shard vessel data into memory."""
        count_vessel = 0
        shard_vessels = []
        while count_vessel < self.shard_size:
            # Varify schedule is sufficient
            if self.cur_schedule_index >= len(self.filter_schedules):
                self._reset_schedules()
            dataset_index, episode_index, step_indices = self.filter_schedules[self.cur_schedule_index]
            steps_data = self.datasets[dataset_index].get_steps_data(episode_index, step_indices)
            # Transformer language, images and action
            shard_vessels += [self.transformer(step_data) for step_data in steps_data]
            self.cur_schedule_index += 1
            count_vessel += len(step_indices)
        return shard_vessels


    def start_load_data(self):
        """Start loading shard vessel according thread."""
        assert self._executor is not None
        self._cache_job = self._executor.submit(self._shard_vessel,)

    def wait_load_data(self):
        """Wait until shard vessels are loaded."""
        assert self._cache_job is not None, "Program system is not initialized."
        self.shard_vessels = self._cache_job.result()
        self._cache_job = None

    def delete_shard_vessels(self):
        """Delete shard vessels."""
        del self.shard_vessels

    def __iter__(self):
        """Get iterator for ShardCacheDataset."""
        assert self.transformer is not None, "ShardCache requires transformer to be initialized."
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.start_load_data()
        while True:
            start_time = time.time()
            self.wait_load_data()
            end_time = time.time()
            print(f"[Data load time]: {end_time - start_time:.2f} seconds", )

            # Immediately load next data
            self.start_load_data()
            for step_data in self.shard_vessels:
                yield step_data
            self.delete_shard_vessels()


if __name__ == "__main__":
    dataset = ShardCacheDataset(
        dataset_paths = "/home/wsj/Desktop/code/VLA/SNOW/datasets/test,"
                        "/home/wsj/Desktop/code/VLA/SNOW/datasets/amass",
        modality_id="YmBot",
        shard_size=2**10,
        vessel_length=2**6,
    )
    dataset.set_transform(lambda x: x)
    for data in dataset:
        x = 1
