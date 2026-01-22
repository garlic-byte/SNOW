import numpy as np

from snow.config import ROBOT_CONFIG


class ActionProcessor:
    def __int__(
        self,
        modality_id: str = None,
        statistics: dict = None,
    ):
        self.modality_id = modality_id
        self.robot_modality = ROBOT_CONFIG[modality_id]
        self.statistics = statistics


    def encoder(self, action: dict[str, np.ndarray]):
        """Normalize value of action into 0-1"""
        for action_key, action_value in action.items():
            assert action_key in self.statistics, f"{action_key} not in statistics."
            params = self.statistics[action_key]
            
