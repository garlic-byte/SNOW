import os

from sympy.printing.pytorch import torch

from snow.config import SnowConfig
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen3VLModel, PreTrainedModel, AutoConfig, \
    AutoModel, BatchFeature

from snow.utils import print_logging

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

        print_logging(f"[Model loaded] Tune backbone vision: {tune_visual}")
        print_logging(f"[Model loaded] Tune backbone language: {tune_llm}")
        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print_logging(f"[Model loaded] Backbone model total params: {total_params:,}")
        print_logging(f"[Model loaded] Backbone model total trainable params: {total_trainable_params:,}, training radio: {total_trainable_params / total_params * 100:.2f}%")

    @staticmethod
    def prepare_input(batch) -> BatchFeature:
        """Prepare inputs for backbone model."""
        use_keys = ['input_ids', 'attention_mask', 'image_grid_thw']
        backbone_input = {key: batch[key] for key in use_keys}
        # Reshape pixel_values for broadcasting and concatenating
        backbone_input["pixel_values"] = batch["pixel_values"].view(-1, batch["pixel_values"].shape[-1])
        return BatchFeature(data=backbone_input)

    def forward(self, backbone_input):
        hidden_features = self.model(**backbone_input)[0]  # shape (batch_size, seq_len, 2048)
        return BatchFeature(
            data={
            "backbone_features": hidden_features,
            "backbone_attention_mask": backbone_input["attention_mask"] == 1,
            "image_mask": backbone_input["input_ids"] == self.model.config.image_token_id  # 151655,
        })

