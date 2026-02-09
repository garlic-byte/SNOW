import os
from PIL import Image
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
import torch

from snow.model.snow_model import SnowModel
from snow.config import SnowConfig


def preprocess_data(vl_processor_path):
    vl_processor = AutoProcessor.from_pretrained(vl_processor_path)
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
    batch = {}
    batch_size = 2

    backbone_input = preprocess_data(vl_model_path)
    pixel_values_shape = backbone_input["pixel_values"].shape
    image_grid_thw_shape = backbone_input["image_grid_thw"].shape
    backbone_input["pixel_values"] = backbone_input["pixel_values"].view(batch_size, -1, pixel_values_shape[-1])
    backbone_input["image_grid_thw"] = backbone_input["image_grid_thw"].view(batch_size, -1, image_grid_thw_shape[-1])
    batch.update(backbone_input)
    batch.update(
        {
        "action": torch.randn((batch_size, 8, 30), device="cuda:0", dtype=torch.bfloat16),
        "embodiment_id": torch.tensor([0]*batch_size, device="cuda:0", dtype=torch.long),
        "action_mask": torch.ones((batch_size, 8, 30), device="cuda:0", dtype=torch.long),
        }
    )

    return BatchFeature(data=batch)


def create_model():
    config = SnowConfig()
    snow_model = SnowModel(config)
    snow_model = snow_model.to("cuda:0")
    snow_model.set_eval()
    with torch.no_grad():
        outputs = snow_model(**inputs)

    os.makedirs(save_dir, exist_ok=True)

    # save config of model
    config.save_pretrained(save_dir)

    # save model's safetensors
    snow_model.save_pretrained(
        save_dir,
        safe_serialization=True,
        max_shard_size="5GB"
    )

    print("save-success")
    return outputs

def load_model():
    snow_model = SnowModel.from_pretrained(save_dir)
    snow_model = snow_model.to("cuda:0")
    snow_model.set_eval()
    with torch.no_grad():
        outputs = snow_model(**inputs)
    return outputs

if __name__ == "__main__":
    torch.manual_seed(64)
    torch.cuda.manual_seed(64)

    vl_model_path = "/home/wsj/Downloads/weights/qwen25-vl-3b"
    img_path = "/home/wsj/Downloads/weights/test_weigths_code/input1.png"
    save_dir = "../weights_qwen25vl"

    inputs = get_inputs()

    # o1 = create_model()
    # print(o1['loss'].item())
    o2 = load_model()
    print(o2['loss'].item())