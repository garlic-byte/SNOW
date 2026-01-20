from dataclasses import dataclass

@dataclass
class ModalityConfig:
    modality_keys: list[str]
    """Keys of modality which contain type of the robot's parts."""

    delta_indices: list[int]
    """Indices of timesteps representing modality changes."""


ROBOT_CONFIG = {
    "YmBot": {
        "observation.images":
            ModalityConfig(
                modality_keys=["front"],
                delta_indices=[0],
            ),
        "language":
            ModalityConfig(
                modality_keys=["task"],
                delta_indices=[0],
            ),
        "action":
            ModalityConfig(
                modality_keys=["root_pos", "root_rot", "dof_pos"],
                delta_indices=list(range(8)),
            ),
    }
}