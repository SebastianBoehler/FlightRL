"""Industrial RGB-D velocity actor; deliberately distinct from forest CTBR actors."""

import hashlib
import numpy as np
import torch
from torch import nn
from .sensing import Observation

CONTRACT = "flightrl.industrial.rgbd128x96.proprio9.velocity_yaw.v1"
LIMITS = np.array([0.4, 0.35, 0.35, 0.3], np.float32)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(24 * 12 * 16, 128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(137, 96), nn.ReLU(), nn.Linear(96, 5))

    def forward(self, image, proprio):
        return self.head(torch.cat([self.encoder(image), proprio], -1))


def images(rgb, depth):
    return np.concatenate(
        [
            rgb.astype(np.float32).transpose(0, 3, 1, 2) / 255,
            depth[:, None].astype(np.float32) / 8,
        ],
        axis=1,
    )


class Policy:
    def __init__(self, path):
        torch.set_num_threads(2)
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        if checkpoint["contract"] != CONTRACT:
            raise ValueError("Incompatible industrial actor contract")
        self.model = Network().eval()
        self.model.load_state_dict(checkpoint["model"])
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    @torch.inference_mode()
    def __call__(self, observation: Observation):
        rgb, depth, proprio = observation.rgb, observation.depth, observation.proprio
        if (
            rgb.shape != (96, 128, 3)
            or depth.shape != (96, 128)
            or proprio.shape != (9,)
        ):
            raise ValueError(
                "Industrial actor expects RGB-D 128x96 and 9 proprioceptive channels"
            )
        if not np.isfinite(depth).all() or not np.isfinite(proprio).all():
            raise ValueError("Nonfinite actor observation")
        out = self.model(
            torch.from_numpy(images(rgb[None], depth[None])),
            torch.from_numpy(proprio[None]),
        ).numpy()[0]
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite actor output")
        return np.clip(out[:4], -1, 1) * LIMITS, float(1 / (1 + np.exp(-out[4])))
