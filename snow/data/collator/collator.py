import torch
from transformers import AutoProcessor, BatchFeature


class DataCollator:
    def __init__(self, processor_path: str):
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, batch_inputs: list) -> BatchFeature:
        """Apply the collation function to the batch."""

        batch = {}
        # Apply version-language processor
        batch["backbone_input"] = self.processor(
            text=[inputs["text"] for inputs in batch_inputs],
            images=[inputs["image"] for inputs in batch_inputs],
            return_tensors="pt",
            padding=True
        )

        # Merge action and embodiment_id to batch
        batch["action_input"] = {
            "action": torch.stack([inputs["action"] for inputs in batch_inputs], dim=0),
            "embodiment_id": torch.cat([inputs["embodiment_id"] for inputs in batch_inputs], dim=0),
            "action_mask": torch.stack([inputs["action_mask"] for inputs in batch_inputs], dim=0),
        }
        return BatchFeature(data={"inputs": batch})