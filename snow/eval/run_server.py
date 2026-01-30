from dataclasses import dataclass
from snow.policy import SnowPolicy

@dataclass
class ServerConfig:
    model_path: str

    modality_id = "YmBot"

    host: str = "localhost"

    port: int = 8080


def run_server(config: ServerConfig):
    SnowPolicy(model_path=config.model_path, modality_id=config.modality_id)