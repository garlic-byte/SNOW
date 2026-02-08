import os
from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Union

import torch
from transformers import PretrainedConfig


@dataclass
class SnowConfig(PretrainedConfig):
    """Unified configuration for Snow model with backbone and action head."""

    # Model identification
    model_type: str = "Snow"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility
    model_path: str = None
    create_mode: bool = True

    # backbone configuration
    model_name: str = "qwen25-vl" # "qwen3-vl"
    backbone_model_path: str = "/home/wsj/Downloads/weights/qwen25-vl-3b" # "/home/wsj/Downloads/weights/qwen3-vl-2b"
    tune_top_llm_layers: int = 4  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 2048  # project_to_dim
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 16

    # Action head configuration parameters
    # max_state_dim: int = 30  # Default from state_shape
    max_action_dim: int = 30  # Default from action_shape
    action_horizon: int = 8
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # Global parameters from YAML
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    # Diffusion model type selection
    use_alternate_vl_dit: bool = True  # True for AlternateVLDiT, False for DiT
    attend_text_every_n_blocks: int = 2

    # Diffusion model configuration with 32 layers (main difference from N15)
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 32,  # 32 layers instead of 16
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # State Augmentation parameters
    state_dropout_prob: float = 0.0  # State dropout probability
    state_additive_noise_scale: float = 0.0  # Scale for additive Gaussian noise on state features

    # Multi-embodiment parameters
    max_num_embodiments: int = 32


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            # PATCH: Backward compatibility for legacy argument "collator_overwrite_image_inputs"
            if key == "collator_overwrite_image_inputs":
                setattr(self, "snow_collator", value)
            # /PATCH
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())

