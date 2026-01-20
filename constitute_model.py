from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import time
import torch


def constitute_version_language_mudules(model_path):
    save_layer_lengths = 12
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda:0"
    )
    qwen3vl = model.model
    language_model = qwen3vl.language_model

    while len(language_model.layers) >= save_layer_lengths:
        language_model.layers.pop(-1)

    return qwen3vl

def constitute_flow_match_mudules():
    pass

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

def get_outputs(inputs, model):
    hidden_features = model(**inputs)
    print(hidden_features)



if __name__ == "__main__":
    vl_model_path = "/home/wsj/Downloads/weights/qwen3-vl-2b"
    inputs = preprocess_data(vl_model_path)
    vl_model = constitute_version_language_mudules(vl_model_path)
    get_outputs(inputs, vl_model)
