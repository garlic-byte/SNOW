from typing import Tuple
from transformers import AutoProcessor
from snow.data import ImageProcessor

class Transformer:
    def __init__(
        self,
        processor_path,
        image_shape: tuple[int, int] | None = None,
        crop_fraction: float = 0.95,
        image_resize: tuple[int, int] | None = None,
        color_jitter: bool = True,
    ):
        self.processor_path = processor_path
        self.version_language_processor = AutoProcessor.from_pretrained(processor_path)

        self.image_processor = ImageProcessor(
            image_shape=image_shape,
            image_resize=image_resize,
            crop_fraction=crop_fraction,
            color_jitter=color_jitter,
        )

    def train(self):
        """Set mode for training and evaluation."""
        self.image_processor.train()

    def eval(self):
        """Set mode for training and evaluation."""
        self.image_processor.eval()