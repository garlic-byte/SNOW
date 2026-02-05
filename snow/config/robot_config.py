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
                modality_keys=["dof_pos"],
                delta_indices=list(range(8)),
            ),
    },
    "libero_panda": {
        "observation.images": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image", "wrist_image"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(8)),
            modality_keys=["arm", "gripper"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["task"],
        )
    },
    "YmBot_D": {
        "observation.images":
            ModalityConfig(
                modality_keys=["top", "middle"],
                delta_indices=[0],
            ),
        "language":
            ModalityConfig(
                modality_keys=["task"],
                delta_indices=[0],
            ),
        "action":
            ModalityConfig(
                modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
                delta_indices=list(range(8)),
            ),
    },
}