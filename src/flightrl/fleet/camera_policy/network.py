"""One role-conditioned RGB-D actor with direct CTBR and visual-report heads."""

import numpy as np
import torch
from torch import nn
from .sensors import CameraPacket


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
            nn.Linear(24 * 6 * 8, 96),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(96 + 18, 96), nn.Tanh(), nn.Linear(96, 5))

    def forward(self, image, state):
        return self.head(torch.cat([self.encoder(image), state], -1))


def tensors(packet, device="cpu"):
    packet.validate()
    image = np.concatenate(
        [
            packet.rgb.astype(np.float32).transpose(0, 3, 1, 2) / 255,
            np.minimum(packet.depth[:, None], 8) / 8,
        ],
        axis=1,
    )
    state = np.c_[packet.proprio, packet.role, packet.messages]
    return torch.tensor(image, device=device), torch.tensor(state, device=device)


class Policy:
    def __init__(self, path):
        torch.set_num_threads(2)
        self.model = Network()
        self.model.load_state_dict(torch.load(path, weights_only=True)["model"])
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, packet: CameraPacket):
        result = self.model(*tensors(packet)).numpy()
        return np.ascontiguousarray(np.clip(result[:, :4], -1, 1), np.float32), 1 / (
            1 + np.exp(-result[:, 4])
        )
