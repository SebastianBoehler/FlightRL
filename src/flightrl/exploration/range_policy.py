from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from flightrl.bounded_action import BoundedNormal

from .range_contract import RANGE_ACTION_DIM, RANGE_EXPLORATION_OBSERVATION_DIM


_ACTION_LOW = (0.0, -1.0)
_ACTION_HIGH = (1.0, 1.0)


class RangeExplorationActorCritic(nn.Module):
    def __init__(self, *, hidden_size: int = 64) -> None:
        super().__init__()
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("range policy hidden size must be positive")
        self.hidden_size = hidden_size
        self.map_encoder = nn.Sequential(
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256, 48),
            nn.SiLU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(10, 16),
            nn.SiLU(),
        )
        self.gated_encoder = nn.GRUCell(64, hidden_size)
        self.actor = nn.Linear(hidden_size, RANGE_ACTION_DIM)
        self.critic = nn.Linear(hidden_size, 1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward_step(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        location, value = self._location_value(observation)
        distribution = BoundedNormal(
            location,
            torch.ones_like(location),
            low=_ACTION_LOW,
            high=_ACTION_HIGH,
        )
        return distribution.mode, value

    def distribution(
        self,
        observation: torch.Tensor,
        action_std: float,
    ) -> tuple[BoundedNormal, torch.Tensor]:
        if not np.isfinite(action_std) or action_std <= 0.0:
            raise ValueError("range policy action standard deviation must be positive")
        location, value = self._location_value(observation)
        scale = torch.full_like(location, float(action_std))
        return (
            BoundedNormal(location, scale, low=_ACTION_LOW, high=_ACTION_HIGH),
            value,
        )

    def _location_value(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(observation)
        map_value = observation[:, :4096].reshape(-1, 4, 32, 32)
        scalars = observation[:, 4096:]
        encoded = torch.cat(
            (self.map_encoder(map_value), self.scalar_encoder(scalars)), dim=1
        )
        hidden = self.gated_encoder(
            encoded,
            torch.zeros(
                (len(encoded), self.hidden_size),
                dtype=encoded.dtype,
                device=encoded.device,
            ),
        )
        return self.actor(hidden), self.critic(hidden).squeeze(1)

    def _validate_inputs(self, observation: torch.Tensor) -> None:
        if observation.ndim != 2 or observation.shape[1] != RANGE_EXPLORATION_OBSERVATION_DIM:
            raise ValueError("range policy observation shape is incompatible")
        if observation.dtype != torch.float32:
            raise ValueError("range policy observation must be float32")
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("range policy observation must be finite")
