from __future__ import annotations

import math

import torch
from torch import nn


VISION_WIDTH = 64
VISION_HEIGHT = 48
VISION_CHANNELS = 3
VISION_PIXELS = VISION_WIDTH * VISION_HEIGHT
VISION_INTENT_DIM = 6
VISION_OBSERVATION_DIM = VISION_CHANNELS * VISION_PIXELS + VISION_INTENT_DIM


def infer_vision_layout(observation_size: int) -> tuple[int, int, int]:
    for privileged_dim in (0, 1):
        visual_values = observation_size - VISION_INTENT_DIM - privileged_dim
        if visual_values <= 0 or visual_values % VISION_CHANNELS:
            continue
        pixels = visual_values // VISION_CHANNELS
        height = math.isqrt(3 * pixels // 4)
        width = pixels // max(height, 1)
        if width * height == pixels and width * 3 == height * 4:
            return width, height, privileged_dim
    raise ValueError(f"visual observation {observation_size} does not encode a 4:3 frame")


def infer_vision_shape(observation_size: int) -> tuple[int, int]:
    width, height, _privileged_dim = infer_vision_layout(observation_size)
    return width, height


class FlightRLVisionEncoder(nn.Module):
    """Compact image-and-intent encoder for the native visual Puffer environment."""

    def __init__(self, observation_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.width, self.height, self.privileged_dim = infer_vision_layout(
            observation_size
        )
        self.vision_dim = VISION_CHANNELS * self.width * self.height
        if self.width * self.height <= 16 * 12:
            vision_features = 32
            self.vision = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.vision_dim, vision_features),
                nn.GELU(),
            )
        else:
            vision_features = 16 * 3 * 4
            self.vision = nn.Sequential(
                nn.Conv2d(VISION_CHANNELS, 8, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((3, 4)),
                nn.Flatten(),
            )
        self.intent = nn.Sequential(nn.Linear(VISION_INTENT_DIM, 8), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(vision_features + 8, hidden_size), nn.GELU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        values = observations.float()
        images = values[:, : self.vision_dim].reshape(
            -1,
            VISION_CHANNELS,
            self.height,
            self.width,
        )
        intent = values[:, self.vision_dim : self.vision_dim + VISION_INTENT_DIM]
        return self.fusion(torch.cat((self.vision(images), self.intent(intent)), dim=1))
