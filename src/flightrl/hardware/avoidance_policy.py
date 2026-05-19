from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn


RANGER_KEYS = ("front", "back", "left", "right", "up", "zrange")


@dataclass(frozen=True, slots=True)
class RangerReading:
    front_m: float
    back_m: float
    left_m: float
    right_m: float
    up_m: float
    zrange_m: float


@dataclass(frozen=True, slots=True)
class AvoidanceCommand:
    vx_m_s: float
    vy_m_s: float
    yawrate_deg_s: float
    zdistance_m: float

    def clipped(self, *, max_speed: float = 0.25, max_yawrate: float = 45.0) -> "AvoidanceCommand":
        return AvoidanceCommand(
            vx_m_s=float(np.clip(self.vx_m_s, -max_speed, max_speed)),
            vy_m_s=float(np.clip(self.vy_m_s, -max_speed, max_speed)),
            yawrate_deg_s=float(np.clip(self.yawrate_deg_s, -max_yawrate, max_yawrate)),
            zdistance_m=float(np.clip(self.zdistance_m, 0.25, 0.8)),
        )


class RangerAvoidancePolicy(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(RANGER_KEYS), hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 4),
        )

    def forward(self, observations):
        tensor = torch.as_tensor(observations, dtype=torch.float32)
        return self.net(tensor)


def normalize_reading(reading: RangerReading, *, max_range_m: float = 4.0) -> np.ndarray:
    return np.asarray(
        [
            _norm(reading.front_m, max_range_m),
            _norm(reading.back_m, max_range_m),
            _norm(reading.left_m, max_range_m),
            _norm(reading.right_m, max_range_m),
            _norm(reading.up_m, max_range_m),
            _norm(reading.zrange_m, max_range_m),
        ],
        dtype=np.float32,
    )


def reading_from_telemetry(values: Mapping[str, float]) -> RangerReading:
    return RangerReading(
        front_m=_range_m(values, "range.front"),
        back_m=_range_m(values, "range.back"),
        left_m=_range_m(values, "range.left"),
        right_m=_range_m(values, "range.right"),
        up_m=_range_m(values, "range.up"),
        zrange_m=_range_m(values, "range.zrange"),
    )


def teacher_command(
    reading: RangerReading,
    *,
    min_distance_m: float = 0.6,
    target_height_m: float = 0.45,
    max_speed_m_s: float = 0.25,
) -> AvoidanceCommand:
    forward_pressure = _pressure(reading.back_m, min_distance_m) - _pressure(reading.front_m, min_distance_m)
    left_pressure = _pressure(reading.right_m, min_distance_m) - _pressure(reading.left_m, min_distance_m)
    height = target_height_m + 0.35 * _pressure(reading.zrange_m, 0.35) - 0.25 * _pressure(reading.up_m, 0.7)
    command = AvoidanceCommand(
        vx_m_s=max_speed_m_s * forward_pressure,
        vy_m_s=max_speed_m_s * left_pressure,
        yawrate_deg_s=0.0,
        zdistance_m=height,
    )
    return command.clipped(max_speed=max_speed_m_s)


def command_from_model(model: RangerAvoidancePolicy, reading: RangerReading) -> AvoidanceCommand:
    obs = normalize_reading(reading)[None, :]
    with torch.no_grad():
        raw = model(obs).squeeze(0).cpu().numpy()
    return AvoidanceCommand(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])).clipped()


def command_array(command: AvoidanceCommand) -> np.ndarray:
    return np.asarray([command.vx_m_s, command.vy_m_s, command.yawrate_deg_s, command.zdistance_m], dtype=np.float32)


def sample_readings(count: int, rng: np.random.Generator) -> list[RangerReading]:
    samples = []
    for _ in range(count):
        values = rng.uniform(0.2, 3.2, size=len(RANGER_KEYS))
        values[5] = rng.uniform(0.25, 0.8)
        samples.append(RangerReading(*[float(v) for v in values]))
    return samples


def _norm(value: float, max_range_m: float) -> float:
    return float(np.clip(value / max_range_m, 0.0, 1.0))


def _pressure(distance_m: float, min_distance_m: float) -> float:
    return float(np.clip((min_distance_m - distance_m) / min_distance_m, 0.0, 1.0))


def _range_m(values: Mapping[str, float], key: str) -> float:
    try:
        raw = float(values.get(key, 4000.0))
    except (TypeError, ValueError):
        raw = 4000.0
    if raw >= 32000.0:
        return 4.0
    return raw / 1000.0
