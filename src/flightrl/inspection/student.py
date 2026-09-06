"""Compact recurrent visual local controller; mission planning remains explicit."""

import torch
from torch import nn


class VisualController(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 12, 5, 3),
            nn.ReLU(),
            nn.Conv2d(12, 16, 3, 2),
            nn.ReLU(),
            nn.AvgPool2d((3, 4)),
            nn.Flatten(),
        )
        self.memory = nn.GRU(64 + 11, 48, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(48, 32), nn.Tanh(), nn.Linear(32, 4), nn.Tanh()
        )

    def forward(self, image, proprio, hidden=None):
        batch, time = image.shape[:2]
        encoded = self.encoder(image.flatten(0, 1)).reshape(batch, time, -1)
        latent, hidden = self.memory(torch.cat((encoded, proprio), -1), hidden)
        return self.head(latent), hidden


def image_tensor(rgb, depth, device):
    rgb = torch.as_tensor(rgb, device=device).float() / 255
    depth = torch.as_tensor(depth, device=device).float().clamp(0, 8) / 8
    return torch.cat((rgb.permute(0, 3, 1, 2), depth[:, None]), 1)


class StudentPolicy:
    def __init__(self, path, device="cpu"):
        self.device = device
        data = torch.load(path, map_location=device, weights_only=True)
        self.model = VisualController().to(device)
        self.model.load_state_dict(data["model"])
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, rgb, depth, proprio, hidden):
        image = image_tensor(rgb[None], depth[None], self.device)[:, None]
        state = torch.as_tensor(proprio, device=self.device)[None, None]
        command, hidden = self.model(image, state, hidden)
        return command[0, 0].cpu().numpy(), hidden
