from typing import Dict
from abc import ABC, abstractmethod

import numpy as np
import torch

from snow.config import SnowConfig, DataConfig
from snow.data.collator.collator import DataCollator
from snow.data.transformer.transformer import Transformer
from snow.model.snow_model import SnowModel
from snow.utils import read_configs


class BasePolicy(ABC):
    def __init__(self, model_path: str):
        assert model_path is not None, (
            f"model_path and modality_id are required for BasePolicy class"
        )
        self.model_path = model_path
        self.model = None
        self.model_config = None
        self.transformer = None
        self.collate = None


    @abstractmethod
    def get_action(self, inputs: Dict) -> Dict:
        raise NotImplementedError

    def reset(self, options: dict):
        return {}

class SnowPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_pretrain_model()
        self.load_transformer_collate()


    def load_pretrain_model(self):
        """Load pre-trained model from local path."""
        model_config = read_configs(self.model_path, "model_config")
        model_config["tune_llm"] = False
        model_config["tune_visual"] = False
        model_config["tune_top_llm_layers"] = 0
        model_config["tune_projector"] = False
        model_config["tune_diffusion_model"] = False
        model_config["tune_vlln"] = False
        self.model_config = SnowConfig(**model_config)
        self.model = SnowModel.from_pretrained(self.model_path, config=self.model_config).to(device='cuda')
        self.model.set_eval()

    def load_transformer_collate(self):
        """Load transformer and collate function."""
        data_config = DataConfig(
            **read_configs(self.model_path, "data_config")
        )
        self.transformer = Transformer(
            processor_path=data_config.processor_path,
            inter_size=data_config.inter_size,
            crop_fraction=data_config.crop_fraction,
            target_size=data_config.target_size,
            color_jitter=data_config.color_jitter,
            modality_id=data_config.modality_id,
            statistics=read_configs(self.model_path, "stats"),
            max_action_dim=self.model_config.max_action_dim
        )
        self.transformer.eval()
        self.collate = DataCollator(
            processor_path=data_config.processor_path,
        )

    def get_action(self, observation: Dict) -> Dict:
        """
        Get action from observation containing:
        {
            'language':
                {
                    'task': str,
                }
            'observation.images':
                {
                    'front': np.ndarray (height_image, width_image, 3),
                }
        }
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Prepare inputs use trans and collate
        prepared_inputs = self.transformer(observation, has_action=False)
        model_inputs = self.collate([prepared_inputs], has_action=False).to(device=device)

        # Get outputs from model
        model_outputs = self.model.get_action(model_inputs).action_pred

        # Decode action
        predict_actions = self.transformer.decode_action(model_outputs)
        return predict_actions


class SimSnowPolicy(SnowPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action(self, observation: Dict) -> Dict:
        """
        Get action for perfectly adapted to simulation environment.
        :param observation:
            key = 'video.image', np.array shape = (batch_size, 1, 256, 256, 3)
            key = 'video.wrist_image', np.array shape = (batch_size, 1, 256, 256, 3)
            key = 'annotation.human.action.task_description', tuple shape = (batch_size)
        :return:
            key = 'action.x', value.shape = (batch_size, action_dimension, 1)
            key = 'action.y', value.shape = (batch_size, action_dimension, 1)
            key = 'action.z', value.shape = (batch_size, action_dimension, 1)
            key = 'action.roll', value.shape = (batch_size, action_dimension, 1)
            key = 'action.pitch', value.shape = (batch_size, action_dimension, 1)
            key = 'action.yaw', value.shape = (batch_size, action_dimension, 1)
            key = 'action.gripper', value.shape = (batch_size, action_dimension, 1)
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = len(observation['annotation.human.action.task_description'])
        prepared_inputs = []
        for batch_index in range(batch_size):
            step_data = {
                'observation.images':{
                    'video.image': observation['video.image'][batch_index, 0],
                    'video.wrist_image': observation['video.wrist_image'][batch_index, 0],
                },
                "language":{
                    "task": observation['annotation.human.action.task_description'][batch_index],
                }
            }
            prepared_inputs.append(self.transformer(step_data, has_action=False))
        model_inputs = self.collate(prepared_inputs, has_action=False).to(device=device)

        # Get outputs from model
        model_outputs = self.model.get_action(model_inputs).action_pred

        # Decode action
        predict_actions = self.transformer.decode_action(model_outputs)

        # Convert simulator actions
        batch_actions = {
            'action.x': predict_actions['arm'][..., 0][..., np.newaxis],
            'action.y': predict_actions['arm'][..., 1][..., np.newaxis],
            'action.z': predict_actions['arm'][..., 2][..., np.newaxis],
            'action.roll': predict_actions['arm'][..., 3][..., np.newaxis],
            'action.pitch': predict_actions['arm'][..., 4][..., np.newaxis],
            'action.yaw': predict_actions['arm'][..., 5][..., np.newaxis],
            'action.gripper': predict_actions['gripper'][..., np.newaxis],
        }
        return batch_actions