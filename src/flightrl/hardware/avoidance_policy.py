from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn

from .avoidance_ttc import min_horizontal_ttc_s, ttc_escape_pressure


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


def reactive_clearance_command(
    reading: RangerReading,
    *,
    range_rate_m_s: RangerReading | None = None,
    clearance_m: float = 0.45,
    hard_clearance_m: float = 0.10,
    target_height_m: float = 0.45,
    max_speed_m_s: float = 0.25,
    ttc_horizon_s: float = 0.0,
    ttc_hard_s: float = 0.12,
    ttc_gain: float = 1.0,
) -> AvoidanceCommand:
    vx_pressure = _axis_clearance_pressure(reading.back_m, reading.front_m, clearance_m, hard_clearance_m)
    vy_pressure = _axis_clearance_pressure(reading.right_m, reading.left_m, clearance_m, hard_clearance_m)
    escape_vx, escape_vy = _open_space_escape_pressure(reading, clearance_m, hard_clearance_m)
    vx_pressure += escape_vx
    vy_pressure += escape_vy
    if range_rate_m_s is not None and ttc_horizon_s > ttc_hard_s:
        ttc_vx, ttc_vy = ttc_escape_pressure(reading, range_rate_m_s, ttc_horizon_s, ttc_hard_s)
        vx_pressure += ttc_gain * ttc_vx
        vy_pressure += ttc_gain * ttc_vy
    bottom_pressure = _clearance_pressure(reading.zrange_m, 0.35, hard_clearance_m)
    top_pressure = _clearance_pressure(reading.up_m, clearance_m, hard_clearance_m)
    command = AvoidanceCommand(
        vx_m_s=max_speed_m_s * vx_pressure,
        vy_m_s=max_speed_m_s * vy_pressure,
        yawrate_deg_s=0.0,
        zdistance_m=target_height_m + 0.30 * bottom_pressure - 0.25 * top_pressure,
    )
    return _clip_horizontal_norm(command, max_speed_m_s).clipped(max_speed=max_speed_m_s)


def min_horizontal_range_m(reading: RangerReading) -> float:
    return min(reading.front_m, reading.back_m, reading.left_m, reading.right_m)


def command_from_model(
    model: RangerAvoidancePolicy,
    reading: RangerReading,
    *,
    max_speed_m_s: float = 0.25,
    max_yawrate_deg_s: float = 45.0,
) -> AvoidanceCommand:
    obs = normalize_reading(reading)[None, :]
    with torch.no_grad():
        raw = model(obs).squeeze(0).cpu().numpy()
    return AvoidanceCommand(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])).clipped(
        max_speed=max_speed_m_s,
        max_yawrate=max_yawrate_deg_s,
    )


def clip_horizontal_norm(command: AvoidanceCommand, *, max_speed: float, max_yawrate: float = 45.0) -> AvoidanceCommand:
    return _clip_horizontal_norm(command, max_speed).clipped(max_speed=max_speed, max_yawrate=max_yawrate)


def smooth_command(
    command: AvoidanceCommand,
    previous: AvoidanceCommand,
    *,
    alpha: float = 0.35,
    max_speed_step_m_s: float = 0.03,
    max_yawrate_step_deg_s: float = 6.0,
    max_zdistance_step_m: float = 0.04,
) -> AvoidanceCommand:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    blended = AvoidanceCommand(
        vx_m_s=_blend(previous.vx_m_s, command.vx_m_s, alpha),
        vy_m_s=_blend(previous.vy_m_s, command.vy_m_s, alpha),
        yawrate_deg_s=_blend(previous.yawrate_deg_s, command.yawrate_deg_s, alpha),
        zdistance_m=_blend(previous.zdistance_m, command.zdistance_m, alpha),
    )
    return AvoidanceCommand(
        vx_m_s=_slew(previous.vx_m_s, blended.vx_m_s, max_speed_step_m_s),
        vy_m_s=_slew(previous.vy_m_s, blended.vy_m_s, max_speed_step_m_s),
        yawrate_deg_s=_slew(previous.yawrate_deg_s, blended.yawrate_deg_s, max_yawrate_step_deg_s),
        zdistance_m=_slew(previous.zdistance_m, blended.zdistance_m, max_zdistance_step_m),
    )


def command_array(command: AvoidanceCommand) -> np.ndarray:
    return np.asarray([command.vx_m_s, command.vy_m_s, command.yawrate_deg_s, command.zdistance_m], dtype=np.float32)


def command_row(command: AvoidanceCommand) -> dict[str, float]:
    return {
        "vx_m_s": command.vx_m_s,
        "vy_m_s": command.vy_m_s,
        "yawrate_deg_s": command.yawrate_deg_s,
        "zdistance_m": command.zdistance_m,
    }


def vertical_velocity_from_height_error(
    command: AvoidanceCommand,
    reading: RangerReading,
    *,
    gain: float = 0.8,
    max_vertical_speed_m_s: float = 0.18,
) -> float:
    return float(np.clip(gain * (command.zdistance_m - reading.zrange_m), -max_vertical_speed_m_s, max_vertical_speed_m_s))


def vertical_velocity_from_clearance(
    reading: RangerReading,
    *,
    top_clearance_m: float = 0.45,
    bottom_clearance_m: float = 0.35,
    hard_clearance_m: float = 0.10,
    max_vertical_speed_m_s: float = 0.18,
) -> float:
    bottom_pressure = _clearance_pressure(reading.zrange_m, bottom_clearance_m, hard_clearance_m)
    top_pressure = _clearance_pressure(reading.up_m, top_clearance_m, hard_clearance_m)
    return float(np.clip(max_vertical_speed_m_s * (bottom_pressure - top_pressure), -max_vertical_speed_m_s, max_vertical_speed_m_s))


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


def _clearance_pressure(distance_m: float, clearance_m: float, hard_clearance_m: float) -> float:
    if clearance_m <= hard_clearance_m:
        raise ValueError("clearance_m must be greater than hard_clearance_m")
    scaled = np.clip((clearance_m - distance_m) / (clearance_m - hard_clearance_m), 0.0, 1.0)
    return float(np.sqrt(scaled))


def _axis_clearance_pressure(positive_side_m: float, negative_side_m: float, clearance_m: float, hard_clearance_m: float) -> float:
    if positive_side_m <= hard_clearance_m and positive_side_m < negative_side_m:
        return 1.0
    if negative_side_m <= hard_clearance_m and negative_side_m < positive_side_m:
        return -1.0
    return _clearance_pressure(positive_side_m, clearance_m, hard_clearance_m) - _clearance_pressure(
        negative_side_m,
        clearance_m,
        hard_clearance_m,
    )


def _open_space_escape_pressure(reading: RangerReading, clearance_m: float, hard_clearance_m: float) -> tuple[float, float]:
    front_pressure = _clearance_pressure(reading.front_m, clearance_m, hard_clearance_m)
    back_pressure = _clearance_pressure(reading.back_m, clearance_m, hard_clearance_m)
    left_pressure = _clearance_pressure(reading.left_m, clearance_m, hard_clearance_m)
    right_pressure = _clearance_pressure(reading.right_m, clearance_m, hard_clearance_m)
    front_back_confinement = min(front_pressure, back_pressure)
    left_right_confinement = min(left_pressure, right_pressure)
    open_x = _open_space_delta(reading.front_m, reading.back_m, clearance_m)
    open_y = _open_space_delta(reading.left_m, reading.right_m, clearance_m)
    return left_right_confinement * open_x, front_back_confinement * open_y


def _open_space_delta(positive_direction_m: float, negative_direction_m: float, clearance_m: float) -> float:
    return float(np.clip((positive_direction_m - negative_direction_m) / clearance_m, -1.0, 1.0))


def _clip_horizontal_norm(command: AvoidanceCommand, max_speed_m_s: float) -> AvoidanceCommand:
    vector = np.asarray([command.vx_m_s, command.vy_m_s], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= max_speed_m_s or norm <= 1e-9:
        return command
    scaled = vector * (max_speed_m_s / norm)
    return AvoidanceCommand(float(scaled[0]), float(scaled[1]), command.yawrate_deg_s, command.zdistance_m)


def _blend(previous: float, target: float, alpha: float) -> float:
    return float(previous + alpha * (target - previous))


def _slew(previous: float, target: float, max_step: float) -> float:
    if max_step < 0.0:
        raise ValueError("max_step must be non-negative")
    return float(previous + np.clip(target - previous, -max_step, max_step))


def _range_m(values: Mapping[str, float], key: str) -> float:
    try:
        raw = float(values.get(key, 4000.0))
    except (TypeError, ValueError):
        raw = 4000.0
    if raw >= 32000.0:
        return 4.0
    return raw / 1000.0
