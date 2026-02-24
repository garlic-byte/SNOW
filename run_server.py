from dataclasses import dataclass

import numpy as np

from snow.eval.zmq_server import PolicyServer
from snow.policy import SnowPolicy, SimSnowPolicy


def generate_random_inputs():
    batch_size  = 2
    img_shape = (batch_size, 1, 256, 256, 3)
    video_image = np.random.randint(low=0,high=256,size=img_shape,dtype=np.uint8)
    video_wrist_image = np.random.randint(low=0,high=256,size=img_shape,dtype=np.uint8)
    language_tuple = tuple(
        f"random_instruction_{np.random.randint(1000, 9999)}"
        for _ in range(batch_size)
    )
    observation = {
        "video.image": video_image,
        "video.wrist_image": video_wrist_image,
        "language": language_tuple
    }
    return observation


@dataclass
class ServerConfig:
    """Configuration class for the server."""
    model_path: str = "outputs/x-0210/gpus_1_batch_size_224" # "outputs/x-0210/gpus_1_batch_size_224"
    """The path of the trained model."""

    host: str = "localhost"
    """The host of the server."""

    port: int = 8000
    """The port of the server."""


def run_server(config: ServerConfig):
    policy = SnowPolicy(model_path=config.model_path)
    # policy = SimSnowPolicy(model_path=config.model_path)
    # outputs = policy.get_action(generate_random_inputs())
    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")

if __name__ == "__main__":
    cfg = ServerConfig()
    run_server(cfg)