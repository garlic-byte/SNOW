import json
import os
import numpy as np
from dataclasses import asdict, is_dataclass
import torch.distributed as dist
import torch

from snow.config import ModalityConfig
from pathlib import Path

GLOBAL_RANK = 0

def serialize_for_json(obj):
    """
    Recursively process all non-JSON-serializable objects,
    automatically converting them to serializable basic types/strings.

    Supported conversions:
    - Numpy types (dtype/ndarray/int/float)
    - Path objects
    - torch.device objects
    - Enum types
    - All unsupported types are converted to string format to eliminate serialization errors!
    """
    # Handle numpy dtype types (root cause of your current error)
    if isinstance(obj, np.dtype):
        return str(obj)
    # Handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle all numpy numeric types (np.int64/np.float32, etc.) → convert to native Python types
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.integer):
        # Convert numpy integers to Python int
        return int(obj)
    elif isinstance(obj, np.floating):
        # Convert numpy floats to Python float
        return float(obj)
    elif isinstance(obj, np.bool_):
        # Convert numpy bool to Python bool
        return bool(obj)
    # Handle dictionaries with recursive traversal (supports nested dictionaries)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    # Handle lists/tuples with recursive traversal
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(i) for i in obj]
    elif isinstance(obj, (int, float)):
        return obj
    elif isinstance(obj, ModalityConfig):
        return serialize_for_json(asdict(obj))
    # Return native JSON-supported types unchanged
    else:
        return str(obj)

def initialize_dist():
    """"""
    global GLOBAL_RANK
    if dist.is_initialized():
        GLOBAL_RANK = dist.get_rank()
    elif "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        # only meaningful for torchrun, for ray it is always 0
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        GLOBAL_RANK = dist.get_rank()

def print_logging(content: str):
    """Print logging messages to stdout only for rank=0"""
    global GLOBAL_RANK
    if GLOBAL_RANK == 0:
        print(content)

def save_dataclass(save_dir, **kwargs):
    """Save dataclass instance to disk"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for class_name, class_instance in kwargs.items():
        if is_dataclass(class_instance):
            class_instance = asdict(class_instance)
        class_instance = serialize_for_json(class_instance)
        with open(save_dir / (class_name + '.json'), "w", encoding="utf-8") as f:
            json.dump(class_instance, f, ensure_ascii=False, indent=4)

def read_configs(model_save_dir, file_name: str):
    """Read json files from disk"""
    model_dir = Path(model_save_dir)
    if not (model_dir / 'configs').exists():
        model_dir = model_dir.parent

    save_dir = model_dir / "configs"
    assert save_dir.exists(), f"There is no config files in {model_dir} or {model_save_dir}"

    content = None
    with open(save_dir / (file_name + '.json'), "r", encoding="utf-8") as f:
        content = json.load(f)
    return content