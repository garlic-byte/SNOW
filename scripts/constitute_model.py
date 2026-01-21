import os
from dataclasses import field

from PIL import Image
from torch import nn
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import time
import torch

from snow.config import SnowConfig
from snow.model.modules import SnowActionHead
from transformers.feature_extraction_utils import BatchFeature
import torch
from safetensors.torch import save_file, load_file

from snow.model.snow_model import SnowModel


def preprocess_data(vl_processor_path):
    vl_processor = AutoProcessor.from_pretrained(vl_processor_path)
    img_path = "/home/wsj/Downloads/weights/test_weigths_code/input1.png"
    pil_image = Image.open(img_path)

    messages = [
        [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                },
                {
                    "type": "image",
                    "image": pil_image,
                },
                {"type": "text", "text": "介绍图片"},
            ],
        }],
        [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                },
                {
                    "type": "image",
                    "image": pil_image,
                },
                {"type": "text", "text": "你是谁?"},
            ],
        }]
    ]

    # Preparation for inference
    inputs = vl_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True
    )
    inputs = inputs.to("cuda:0")
    return inputs


def get_inputs():
    vl_model_path = "/home/wsj/Downloads/weights/qwen3-vl-2b"
    backbone_inputs = preprocess_data(vl_model_path)
    action_input = BatchFeature({
        "action": torch.randn((2, 8, 30), device="cuda:0", dtype=torch.bfloat16),
        "embodiment_id": torch.tensor([0, 0], device="cuda:0", dtype=torch.long),
        "action_mask": torch.ones((2, 8, 30), device="cuda:0", dtype=torch.long),
    })
    return backbone_inputs, action_input


def create_model(backbone_inputs, action_input):
    config = SnowConfig()
    snow_model = SnowModel(config)
    snow_model = snow_model.to("cuda:0")


    outputs = snow_model(backbone_inputs, action_input)

    save_dir = "../weights"
    os.makedirs(save_dir, exist_ok=True)

    # 保存配置
    config.save_pretrained(save_dir)

    # 保存权重为 safetensors
    snow_model.save_pretrained(
        save_dir,
        safe_serialization=True,
        max_shard_size="5GB"
    )

    print("save-success")
    return outputs

def load_model(backbone_inputs, action_input):
    save_dir = "./weights_3"
    snow_model =  SnowModel.from_pretrained(save_dir)
    snow_model = snow_model.to("cuda:0")

    outputs = snow_model(backbone_inputs, action_input)
    return outputs

if __name__ == "__main__":
    torch.manual_seed(64)
    torch.cuda.manual_seed(64)
    backbone_inputs, action_input = get_inputs()

    with torch.no_grad():
        # o1 = create_model(backbone_inputs, action_input)
        # print(o1['loss'].item())
        o2 = load_model(backbone_inputs, action_input)
        print(o2['loss'].item())