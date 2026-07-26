from __future__ import annotations

import torch
from torch import nn


VISION_WIDTH = 64
VISION_HEIGHT = 48
VISION_CHANNELS = 3
VISION_PIXELS = VISION_WIDTH * VISION_HEIGHT
VISION_OBSERVATION_DIM = VISION_CHANNELS * VISION_PIXELS + 6


class FlightRLVisionEncoder(nn.Module):
    """Compact image-and-intent encoder for the native visual Puffer environment."""

    def __init__(self, observation_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        if observation_size != VISION_OBSERVATION_DIM:
            raise ValueError(
                f"visual encoder expected {VISION_OBSERVATION_DIM} observations, got {observation_size}"
            )
        self.vision = nn.Sequential(
            nn.Conv2d(VISION_CHANNELS, 8, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(12, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
        )
        self.intent = nn.Sequential(nn.Linear(6, 16), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(16 * 3 * 4 + 16, hidden_size), nn.GELU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        values = observations.float()
        images = values[:, : VISION_CHANNELS * VISION_PIXELS].reshape(
            -1,
            VISION_CHANNELS,
            VISION_HEIGHT,
            VISION_WIDTH,
        )
        intent = values[:, -6:]
        return self.fusion(torch.cat((self.vision(images), self.intent(intent)), dim=1))
