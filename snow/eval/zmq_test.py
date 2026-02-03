from snow.eval.zmq_client import PolicyClient
import numpy as np


policy = PolicyClient(host="localhost", port=8000) # Connect to the policy server
if not policy.ping(): # Verify connection
    raise RuntimeError("Cannot connect to policy server!")
obs = {
        "observation.images":
            {
                "front": np.zeros((480, 640, 3), dtype=np.uint8),
            },
        "language": {
            "task": "put the white mug on the left plate and put the yellow and white mug on the right plate"
        }
}
action = policy.get_action(obs) # Run inference

for key, value in action.items():
    if isinstance(value, np.ndarray):
        print(key, value.shape)
