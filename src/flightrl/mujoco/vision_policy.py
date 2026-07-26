from __future__ import annotations

import torch
import torch.nn as nn

import pufferlib.pytorch

from flightrl.mujoco.vision_env import INTENT_DIM


class PufferVisionSetpointPolicy(nn.Module):
    def __init__(self, env, hidden_size: int = 96) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = True
        observation_size = int(env.single_observation_space.shape[0])
        self.vision_size = observation_size - INTENT_DIM
        if self.vision_size != 3 * 48 * 64:
            raise ValueError(f"expected a 3x48x64 visual observation, got {self.vision_size} values")
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
        )
        self.intent_encoder = nn.Sequential(nn.Linear(INTENT_DIM, 24), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(24 * 3 * 4 + 24, hidden_size), nn.ReLU())
        self.decoder_mean = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 4), std=0.01)
        self.decoder_logstd = nn.Parameter(torch.full((1, 4), -1.5))
        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        batch = observations.shape[0]
        flat = observations.reshape(batch, -1).float()
        vision = flat[:, : self.vision_size].reshape(batch, 3, 48, 64)
        intent = flat[:, self.vision_size :]
        return self.fusion(torch.cat((self.vision_encoder(vision), self.intent_encoder(intent)), dim=1))

    def decode_actions(self, hidden: torch.Tensor):
        mean = self.decoder_mean(hidden)
        std = torch.exp(self.decoder_logstd.expand_as(mean))
        return torch.distributions.Normal(mean, std), self.value(hidden)

    def forward_eval(self, observations: torch.Tensor, state=None):
        return self.decode_actions(self.encode_observations(observations, state))

    def forward(self, observations: torch.Tensor, state=None):
        return self.forward_eval(observations, state)
