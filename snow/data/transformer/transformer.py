from typing import Any

import numpy as np
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
        max_action_dim=None,
        max_action_horizon=None,
    ):
        self.processor_path = processor_path
        self.version_language_processor = VisionLanguageProcessor(processor_path)
        self.action_modality_keys = ROBOT_CONFIG[modality_id]['action'].modality_keys
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon

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

    def decode_action(self, action: torch.tensor) -> dict[str, np.ndarray]:
        """
        Decode tensor into action with action_modality.
        Args:
            action (torch.tensor): [batch_size, action_horizon, action_dimension]
        Returns:
            {
                action_key: [action_horizon, action_key_dimension]
            }
        """
        return self.action_processor.decoder(action.cpu().detach().to(dtype=torch.float32).numpy())


    def __call__(self, step_data: dict[str, Any], has_action=True) -> dict[str, Any]:
        """
        Transform raw data -> normalized data for Model inputs.
        :param step_data: dictionary mapping modality key -> raw data.
        """
        normalized_data = {}
        # Step 1. normalize images
        normalized_images = self.image_processor(step_data['observation.images'])

        # Step 2. normalize actions
        if has_action:
            normalized_actions = self.action_processor(step_data["action"])
            normalized_actions_torch = torch.cat(
                [
                    torch.from_numpy(normalized_actions[action_key])
                        for action_key in self.action_modality_keys
                ], dim=-1,
            )
            # Padding action to max_action_dim
            action_horizon, action_dimension = normalized_actions_torch.shape
            normalized_actions_torch = torch.cat(
                [
                    normalized_actions_torch,
                    torch.zeros(action_horizon, self.max_action_dim - action_dimension)
                ],dim=-1,
            )
            # Padding action to max_action_horizon
            normalized_actions_torch = torch.cat(
                [
                    normalized_actions_torch,
                    torch.zeros(self.max_action_horizon - action_horizon, self.max_action_dim)
                ],dim=0,
            )
            # Generate mask for action
            action_mask = torch.ones_like(normalized_actions_torch)
            action_mask[:, action_dimension:] = 0
            action_mask[action_horizon:, :] = 0
            normalized_data["action"] = normalized_actions_torch
            normalized_data["action_mask"] = action_mask

        # Step 3. normalize images and languages
        normalized_version_language = self.version_language_processor(normalized_images, step_data["language"]['task'])
        normalized_data.update(normalized_version_language)

        # Step 4. addition embodiment_id
        normalized_data["embodiment_id"] = torch.tensor([0], dtype=torch.long)
        return normalized_data
