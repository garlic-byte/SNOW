from typing import Dict
from abc import ABC, abstractmethod

from snow.config import SnowConfig, DataConfig
from snow.data.collator.collator import DataCollator
from snow.data.transformer.transformer import Transformer
from snow.model.snow_model import SnowModel
from snow.utils import read_configs


class BasePolicy(ABC):
    def __init__(self, model_path: str, modality_id: str):
        assert model_path is not None and modality_id is not None, (
            f"model_path and modality_id are required for BasePolicy class"
        )
        self.model_path = model_path
        self.modality_id = modality_id
        self.model = None
        self.model_config = None
        self.transformer = None
        self.collate = None


    @abstractmethod
    def get_action(self, inputs: Dict) -> Dict:
        raise NotImplementedError


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
            max_action_dim=self.model_config["max_action_dim"]
        )
        self.transformer.eval()
        self.collate = DataCollator(
            processor_path=data_config.processor_path,
        )

    def get_action(self, inputs: Dict) -> Dict:
        """
        Get action from inputs like:
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
        prepared_inputs = self.transformer(inputs)


