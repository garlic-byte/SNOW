from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


class VisionLanguageProcessor:
    """Process language and vision."""

    def __init__(self, processor_path: str):
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, vision: list[Image.Image], language: str):
        """Apply vision-language template."""
        conversation = [
            {
                'role': "user",
                'content': [{"type": "image", "image": image} for image in vision]
                + [{"type": "text", "text": language}],
            }
        ]
        # Apply chat template
        vlm_text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        vlm_image = process_vision_info(conversation)[0]

        return {"image": vlm_image, "text": vlm_text}