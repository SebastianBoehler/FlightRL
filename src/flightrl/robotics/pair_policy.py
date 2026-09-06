"""A compact learned visual-servo policy shared by drone and wheeled rover."""

import hashlib
import numpy as np
import torch
from torch import nn

CONTRACT = "flightrl.marker_depth_proprio21.drone_rover.velocity_yaw.v1"
LIMITS = np.array([0.4, 0.35, 0.35, 0.6], np.float32)


class PairNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(21, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.layers(x)


class PairPolicy:
    controls_both = True

    def __init__(self, path):
        torch.set_num_threads(2)
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        if checkpoint["contract"] != CONTRACT:
            raise ValueError("Incompatible mixed-robot feature policy")
        self.model = PairNetwork().eval()
        self.model.load_state_dict(checkpoint["model"])
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    @torch.inference_mode()
    def act(self, features):
        if features.shape != (21,) or not np.isfinite(features).all():
            raise ValueError("Finite 21-channel visual sensor features required")
        out = self.model(torch.from_numpy(features[None])).numpy()[0]
        if not np.isfinite(out).all():
            raise ValueError("Invalid policy output")
        return np.clip(out, -1, 1) * LIMITS
