import numpy as np
import torch

from snow.config import ROBOT_CONFIG

def normalize_min_max(action: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """
    Normalize min and max of an action into [-1, 1].
    :param action: np.ndarray, shape of (action_horizon, action_dimension)
    :param params: dict[str, np.ndarray]
        param['min']: np.ndarray, shape of (action_dimension)
        param['max']: np.ndarray, shape of (action_dimension)
    """
    min_action = np.array(params['min'], dtype=np.float32)
    max_action = np.array(params['max'], dtype=np.float32)

    normalized_action = np.zeros_like(action)
    # Parameters who is close are not used to calculate
    mask = ~np.isclose(max_action, min_action)
    normalized_action[..., mask] = (action[..., mask] - min_action[..., mask]) / (max_action[..., mask] - min_action[..., mask])

    # Convert to [-1, 1]
    normalized_action = 2 * normalized_action - 1
    return normalized_action


def denormalize_min_max(action: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Recover action from normalized min and max to original range."""
    min_action = np.array(params['min'], dtype=np.float32)
    max_action = np.array(params['max'], dtype=np.float32)

    denormalized_action = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
    denormalized_action = denormalized_action * (max_action - min_action) + min_action
    return denormalized_action



class ActionProcessor:
    def __init__(self, modality_id: str, statistics: dict):
        self.action_modality = ROBOT_CONFIG[modality_id]['action']
        self.statistics = statistics


    def __call__(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Normalize value of action into [-1, 1]
        :param action: Dictionary mapping joint-group -> np.ndarray,
                    shape of (action_horizon, action_dimension)
        :return: dict[str, np.ndarray] same shape as action.
        """
        normalized_action = {}
        for action_key, action_value in action.items():
            assert action_key in self.statistics, f"{action_key} not in statistics."
            params = self.statistics[action_key]
            normalized_action[action_key] = normalize_min_max(action_value, params)

        return normalized_action

    def decoder(self, action: torch.tensor) -> dict[str, np.ndarray]:
        """
        Denormalize value of action into origin range.
        :param action: np.ndarray, shape of (batch_size, action_horizon, action_dimension)
        """
        denormalized_action = {}
        start_index = end_index = 0
        for action_key in self.action_modality.modality_keys:
            params = self.statistics[action_key]
            end_index += len(params['max'])
            denormalized_action[action_key] = denormalize_min_max(
                action[..., start_index: end_index], params
            ).squeeze()
            start_index = end_index

        return denormalized_action
