from transformers import PreTrainedModel, BatchFeature
import torch

from snow.config import SnowConfig
from snow.model.modules import SnowActionHead
from snow.model.modules.backbone_module import SnowBackbone


class SnowModel(PreTrainedModel):
    config_class = SnowConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: SnowConfig):
        super().__init__(config)
        self.backbone = SnowBackbone(config)
        self.action_head = SnowActionHead(config).to(dtype=self.backbone.dtype)

    def forward(self, backbone_inputs, action_input) -> BatchFeature:
        hidden_features = self.backbone(backbone_inputs)[0]  # shape (2, seq_len, 2048)
        image_mask = backbone_inputs["input_ids"] == self.backbone.model.config.image_token_id  # 151655

        backbone_output = BatchFeature({
            "backbone_features": hidden_features,
            "backbone_attention_mask": backbone_inputs["attention_mask"] == 1,
            "image_mask": image_mask,
        })
        batch_output = self.action_head(backbone_output, action_input)

        return batch_output