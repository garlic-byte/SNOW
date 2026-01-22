import time

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
import torch.distributed as dist
from concurrent.futures import Future, ThreadPoolExecutor

from snow.data.datasets import LerobotDataset

class ShardCacheDataset(IterableDataset):
    def __init__(
        self,
        dataset_paths: tuple,
        modality_id: str = None,
        video_backend: str = "ffmpeg",
        video_backend_kwargs: dict = None,
        shard_size: int = 2**10,
        seed: int = 64,
    ):
        super().__init__()
        self.dataset_paths = dataset_paths
        self.shard_size = shard_size
        self.seed = seed

        # Initialize all datasets
        self.datasets = [LerobotDataset(
                dataset_index=dataset_index,
                dataset_path=dataset_path,
                modality_id=modality_id,
                video_backend=video_backend,
                video_backend_kwargs=video_backend_kwargs,
            )for dataset_index, dataset_path in enumerate(self.dataset_paths)]
        self.epoch = -1
        # Initialize schedules (dataset_index, episode_index, step_index)
        self.total_schedules = np.vstack([dataset.initialize_schedule() for dataset in self.datasets], dtype=np.int32)

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

    def _reset_schedules(self):
        """Create random schedules for loading datasets."""
        self.epoch += 1
        self.cur_schedule_index = 0
        rng = np.random.default_rng(self.seed + self.epoch)
        random_schedules = self.total_schedules.copy()
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
        self.filter_schedules = filter_schedules


    def _shard_vessel(self):
        """Load shard vessel data into memory."""
        count_vessel = 0
        shard_vessels = []
        while count_vessel < self.shard_size:
            # Varify schedule is sufficient
            if self.cur_schedule_index >= len(self.filter_schedules):
                self._reset_schedules()
            dataset_index, episode_index, step_index = self.filter_schedules[self.cur_schedule_index]
            shard_vessels.append(self.datasets[dataset_index].get_step_data(episode_index, step_index))
            self.cur_schedule_index += 1
            count_vessel += 1
        self.shard_vessels = shard_vessels


    def start_load_data(self):
        """Start loading shard vessel according thread."""
        assert self._executor is not None
        self._cache_job = self._executor.submit(self._shard_vessel,)

    def wait_load_data(self):
        """Wait until shard vessels are loaded."""
        assert self._cache_job is not None, "Program system is not initialized."
        self._cache_job.result()
        self._cache_job = None


    def __iter__(self):
        """Get iterator for ShardCacheDataset."""
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

if __name__ == "__main__":
    dataset = ShardCacheDataset(
        dataset_paths = (
            "/home/wsj/Desktop/code/VLA/SNOW/datasets/test",
            "/home/wsj/Desktop/code/VLA/SNOW/datasets/amass",
        ),
        modality_id="YmBot",
        shard_size=10,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=4,
        drop_last=True
    )

    print("===== Multiple workers test =====")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx} - info: {batch}")
