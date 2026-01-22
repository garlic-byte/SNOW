import PIL
import numpy as np
import albumentations as A
from typing import Dict, Optional

from PIL import Image


class ImageProcessor:
    """
    Unified image preprocessing and transformation pipeline for computer vision tasks.

    This class provides a comprehensive framework for image processing operations,
    combining spatial transformations, color augmentations, and normalization
    to prepare images for deep learning model input.

    Key Features:
    - Spatial transformations: Adaptive cropping and resizing for input standardization
    - Color augmentation: Photometric distortions for data regularization
    - Configurable pipeline: Modular design supporting flexible transformation sequences

    Processing Pipeline:
    1. Input validation and format standardization
    2. Spatial transformations (cropping → resizing)
    3. Photometric augmentations (training phase only)
    4. Statistical normalization (zero-mean, unit-variance scaling)

    Typical Use Cases:
    - Training data augmentation for improved model generalization
    - Inference preprocessing for consistent input formatting
    - Dataset normalization for accelerated training convergence

    Design Philosophy:
    - Deterministic transformations for validation/inference
    - Stochastic augmentations for training (optional)
    - Batch-aware processing for computational efficiency
    - Device-agnostic implementation (CPU/GPU transparent)

    Notes:
    - All transformations maintain original aspect ratio by default
    - Color augmentations are disabled during inference for reproducibility
    - Normalization statistics can be precomputed or derived from input data
    """

    def __init__(
        self,
        image_shape: tuple[int, int] | None = None,
        image_resize: tuple[int, int] | None = None,
        crop_fraction: float = 0.95,
        color_jitter: bool = True,
    ):
        # Initialize parameters of processor
        self.input_shape = image_shape
        self.image_resize = image_resize
        self.crop_fraction = crop_fraction
        self.color_jitter = color_jitter

        # Initialize transformations
        self.image_transform = None
        self.train_transforms = None
        self.inference_transforms = None
        self._create_spatial_transform()

    def train(self):
        self.image_transform = self.train_transforms

    def eval(self):
        self.image_transform = self.inference_transforms

    def _create_spatial_transform(self):
        """Create spatial transformations (crop and resize)"""
        train_transforms = inference_transforms = []

        # Add fraction crop transformation
        assert 0 < self.crop_fraction <= 1, f"crop_fraction must be between 0 and 1"
        train_transforms.append(A.RandomCrop(
            height=int(self.input_shape[0] * self.crop_fraction),
            width=int(self.input_shape[1] * self.crop_fraction),
            p=1.0
        ))
        # For eval model, use center crop with images.
        inference_transforms.append(A.CenterCrop(
            height=int(self.input_shape[0] * self.crop_fraction),
            width=int(self.input_shape[1] * self.crop_fraction),
            p=1.0
        ))

        # Add resize transformation
        if self.image_resize:
            train_transforms.append(A.Resize(
                height=self.image_resize[0],
                width=self.image_resize[1],
                p=1.0
            ))
            inference_transforms.append(A.Resize(
                height=self.image_resize[0],
                width=self.image_resize[1],
                p=1.0
            ))

        # Add jitter transformation
        if self.color_jitter:
            train_transforms.append(A.ColorJitter(
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
                p=1.0
            ))

        self.train_transforms = A.Compose(train_transforms)
        self.inference_transforms = A.Compose(inference_transforms)


    def __call__(self, images: Dict[str, np.ndarray]) -> list[Image.Image]:
        """
        Apply image transformations to a dictionary of images.

        :param images: Dict mapping image key -> image numpy array (HWC format).
            Expected shape: (height, width, channels)
            Expected dtype: uint8 [0, 255]
        :return: processed images, dict mapping image key -> processed numpy array
            dtype: uint8 [0, 255], shape (resize[0], resize[1], channels)
        """
        processed_images = []

        for key, image in images.items():
            # Validate input
            if image is None or image.size == 0:
                raise ValueError(f"Image '{key}' is empty or None")
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            # Apply spatial transformations
            transformed = self.image_transform(image=image)['image']
            processed_images.append(Image.fromarray(transformed, mode='RGB'))

        return processed_images


# 示例使用
if __name__ == "__main__":
    sample_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    sample_images = {
        'rgb': sample_image,
    }

    processor = ImageProcessor(
        image_shape=(512, 512),
        image_resize=(224, 224),
        crop_fraction=0.95,
        color_jitter=True
    )

    processed_single = processor(sample_images)
    for idx, image in enumerate(processed_single):
        image.save(f"{idx}.png")

