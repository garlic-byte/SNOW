from dataclasses import dataclass

from snow.eval.zmq_server import PolicyServer
from snow.policy import SnowPolicy

@dataclass
class ServerConfig:
    """Configuration class for the server."""
    model_path: str = "/home/wsj/Desktop/code/VLA/SNOW/outputs/x-project/gpus_1_batch_size_448"
    """The path of the trained model."""

    host: str = "localhost"
    """The host of the server."""

    port: int = 8000
    """The port of the server."""


def run_server(config: ServerConfig):
    policy = SnowPolicy(model_path=config.model_path)
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