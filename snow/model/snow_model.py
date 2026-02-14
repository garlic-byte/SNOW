from transformers import PreTrainedModel, BatchFeature
import torch
import tree
import torch.cuda.nvtx as nvtx

from snow.config import SnowConfig
from snow.model.modules import SnowActionHead, DriftActionHead
from snow.model.modules.backbone_module import SnowBackbone


class SnowModel(PreTrainedModel):
    config_class = SnowConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: SnowConfig):
        super().__init__(config)
        self.backbone = SnowBackbone(config)
        self.action_head = SnowActionHead(config).to(dtype=self.backbone.dtype)

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def set_eval(self):
        for p in self.parameters():
            p.requires_grad = False

        print("Start mode of inferencing...")


    def prepare_input(self, inputs, eval_model=False):
        """Change dtype and convert to BatchFeature."""
        def to_dtype(x):
            if torch.is_floating_point(x):
                return x.to(dtype=self.dtype)
            else:
                return x
        batch_backbone_input = self.backbone.prepare_input(inputs)
        batch_action_input = self.action_head.prepare_input(inputs, eval_model=eval_model)

        batch_backbone_input = tree.map_structure(to_dtype, batch_backbone_input)
        batch_action_input = tree.map_structure(to_dtype, batch_action_input)
        return batch_backbone_input, batch_action_input


    def forward(self, **kwargs) -> BatchFeature:
        """Forward pass of the Snow model."""
        backbone_input, action_input = self.prepare_input(kwargs)

        backbone_output = self.backbone(backbone_input)
        batch_output = self.action_head(backbone_output, action_input)
        return batch_output

    def get_action(self, inputs: BatchFeature) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs, eval_model=True)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        return action_outputs