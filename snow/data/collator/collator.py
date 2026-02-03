import torch
from transformers import AutoProcessor, BatchFeature


class DataCollator:
    def __init__(self, processor_path: str):
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, batch_inputs: list, has_action=True) -> BatchFeature:
        """
        Apply the collation function to the batch.
        :param has_action: Whether return an action.
        :return: batch process like:
        {
            "input_ids": torch.tensor shape (B, Seq_len),
            "attention_mask": torch.tensor shape (B, Seq_len),
            "pixel_values": torch.tensor shape (B, C, H, W),
            "image_grid_thw": torch.tensor shape (B, C, H, W),
            "action": torch.tensor shape (B, Action_horizon, Action_dimension)
            "action_mask": torch.tensor shape (B, Action_horizon, Action_dimension),
        }
        """

        batch = {
            "embodiment_id": torch.cat(
                [inputs["embodiment_id"] for inputs in batch_inputs], dim=0
            )
        }
        batch_size = len(batch_inputs)
        # Apply version-language processor
        backbone_input = self.processor(
            text=[inputs["text"] for inputs in batch_inputs],
            images=[inputs["image"] for inputs in batch_inputs],
            return_tensors="pt",
            padding=True
        )

        # Reshape pixel_values and image_grid_thw for broadcasting and concatenating
        pixel_values_shape = backbone_input["pixel_values"].shape
        image_grid_thw_shape = backbone_input["image_grid_thw"].shape
        backbone_input["pixel_values"] = backbone_input["pixel_values"].view(batch_size, -1, pixel_values_shape[-1])
        backbone_input["image_grid_thw"] = backbone_input["image_grid_thw"].view(batch_size, -1, image_grid_thw_shape[-1])
        batch.update(backbone_input)

        # Merge action and embodiment_id to batch
        if has_action:
            action_input = {
                "action": torch.stack([inputs["action"] for inputs in batch_inputs], dim=0),
                "action_mask": torch.stack([inputs["action_mask"] for inputs in batch_inputs], dim=0),
            }
            batch.update(action_input)
        return BatchFeature(data=batch)