from typing import Any

import torch

from snow.config import ROBOT_CONFIG
from snow.data import ImageProcessor, ActionProcessor, VisionLanguageProcessor

class Transformer:
    def __init__(
        self,
        processor_path: str,
        inter_size: tuple[int, int] | None = None,
        crop_fraction: float = 0.95,
        target_size: tuple[int, int] | None = None,
        color_jitter: bool = True,
        modality_id: str = None,
        statistics: dict = None,
    ):
        self.processor_path = processor_path
        self.version_language_processor = VisionLanguageProcessor(processor_path)
        self.action_modality_keys = ROBOT_CONFIG[modality_id]['action'].modality_keys

        self.image_processor = ImageProcessor(
            inter_size=inter_size,
            target_size=target_size,
            crop_fraction=crop_fraction,
            color_jitter=color_jitter,
        )

        self.action_processor = ActionProcessor(
            modality_id=modality_id,
            statistics=statistics,
        )

    def train(self):
        """Set mode for training and evaluation."""
        self.image_processor.train()

    def eval(self):
        """Set mode for training and evaluation."""
        self.image_processor.eval()

    def __call__(self, step_data: dict[str, Any]) -> dict[str, Any]:
        """
        Transform raw data -> normalized data for Model inputs.
        :param step_data: dictionary mapping modality key -> raw data.
        """
        normalized_data = {}
        # Step 1. normalize images
        normalized_images = self.image_processor(step_data['observation.images'])

        # Step 2. normalize actions
        normalized_actions = self.action_processor(step_data["action"])
        normalized_actions_torch = torch.cat(
            [
                torch.from_numpy(normalized_actions[action_key])
                    for action_key in self.action_modality_keys
            ], dim=-1,
        )
        normalized_data["action"] = normalized_actions_torch

        # Step 3. normalize images and languages
        normalized_version_language = self.version_language_processor(normalized_images, step_data["language"])
        normalized_data.update(normalized_version_language)

        # Step 4. addition embodiment_id
        normalized_data["embodiment_id"] = torch.tensor([0], dtype=torch.long)

        # Step 5. addition action_mask
        normalized_data["action_mask"] = torch.ones_like(normalized_actions_torch)
        return normalized_data
