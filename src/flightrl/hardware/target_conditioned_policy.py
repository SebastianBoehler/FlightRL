from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from flightrl.hardware.avoidance_policy import AvoidanceCommand, RangerReading, normalize_reading


@dataclass(frozen=True, slots=True)
class TargetSpec:
    direction_deg: float
    speed_m_s: float


class TargetConditionedPolicy(nn.Module):
    def __init__(self, hidden_size: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 4),
        )

    def forward(self, observations):
        tensor = torch.as_tensor(observations, dtype=torch.float32)
        return self.net(tensor)


def target_observation(reading: RangerReading, target: TargetSpec, *, max_speed_m_s: float = 1.1) -> np.ndarray:
    angle = radians(target.direction_deg)
    return np.concatenate(
        [
            normalize_reading(reading),
            np.asarray([cos(angle), sin(angle), np.clip(target.speed_m_s / max_speed_m_s, 0.0, 1.0)], dtype=np.float32),
        ]
    ).astype(np.float32)


def command_from_target_model(
    model: TargetConditionedPolicy,
    reading: RangerReading,
    target: TargetSpec,
    *,
    max_speed_m_s: float = 0.55,
    max_yawrate_deg_s: float = 45.0,
) -> AvoidanceCommand:
    observation = target_observation(reading, target)[None, :]
    with torch.no_grad():
        raw = model(observation).squeeze(0).cpu().numpy()
    return AvoidanceCommand(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])).clipped(
        max_speed=max_speed_m_s,
        max_yawrate=max_yawrate_deg_s,
    )


def load_target_policy(path: str | Path) -> TargetConditionedPolicy:
    checkpoint = torch.load(path, map_location="cpu")
    model = TargetConditionedPolicy(hidden_size=int(checkpoint.get("hidden_size", 96)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
