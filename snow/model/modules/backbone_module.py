import os

from sympy.printing.pytorch import torch

from snow.config import SnowConfig
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen3VLModel, PreTrainedModel, AutoConfig, AutoModel


class SnowBackbone(PreTrainedModel):
    def __init__(self, config: SnowConfig):
        super().__init__(config)
        # Load model from local directory
        self.config = config
        extra_kwargs = {}
        if config.model_dtype == "float16":
            extra_kwargs["dtype"] = torch.float16
        elif config.model_dtype == "bfloat16":
            extra_kwargs["dtype"] = torch.bfloat16

        if config.model_name == "qwen3-vl":
            qwen_config_path = os.path.join(os.path.dirname(__file__), "qwen3-vl")
            qwen_config = AutoConfig.from_pretrained(qwen_config_path, trust_remote_code=True)
            qwen_config.vision_config.update(extra_kwargs)
            self.model = Qwen3VLModel(qwen_config)
            # self.model = Qwen3VLForConditionalGeneration.from_pretrained(config.backbone_model_path, **extra_kwargs).model
        else:
            raise NotImplementedError(f"{config.model_name} is not supported.")

        # Reduce partly modules from model
        while len(self.model.language_model.layers) > config.select_layer:
            self.model.language_model.layers.pop(-1)

        # Set partly trained parameters
        self.set_trained_params(config.tune_llm, config.tune_visual, config.tune_top_llm_layers)

    def set_trained_params(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int = None):
        # Close all parameters gradient
        for param in self.model.parameters():
            param.requires_grad = False

        # Open partly parameters gradient
        if tune_llm:
            for param in self.model.language_model.parameters():
                param.requires_grad = True
        if tune_visual:
            for param in self.model.visual.parameters():
                param.requires_grad = True
        if tune_top_llm_layers:
            for layer in self.model.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model loaded] Total params: {total_params:,}")
        print(f"[Model loaded] Total trainable params: {total_trainable_params:,}, training radio: {total_trainable_params / total_params * 100:.2f}%")


    def forward(self, backbone_inputs):
        return self.model(**backbone_inputs)