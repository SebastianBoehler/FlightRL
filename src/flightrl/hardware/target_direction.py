from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np

from flightrl.hardware.avoidance_ttc import time_to_collision_s
from flightrl.hardware.avoidance_policy import (
    AvoidanceCommand,
    RangerReading,
    min_horizontal_range_m,
    reactive_clearance_command,
)


@dataclass(frozen=True, slots=True)
class TargetDirectionConfig:
    direction_deg: float = 0.0
    target_speed_m_s: float = 0.20
    clearance_m: float = 1.30
    hard_clearance_m: float = 0.10
    target_height_m: float = 0.50
    avoidance_speed_m_s: float = 0.95
    max_speed_m_s: float = 0.95
    slowdown_gain: float = 0.85
    avoidance_gain: float = 1.0
    ttc_horizon_s: float = 0.0
    ttc_hard_s: float = 0.12
    ttc_gain: float = 1.0


def target_direction_command(
    reading: RangerReading,
    config: TargetDirectionConfig,
    *,
    range_rate_m_s: RangerReading | None = None,
) -> AvoidanceCommand:
    cruise = cruise_vector(config.direction_deg, config.target_speed_m_s)
    avoidance = reactive_clearance_command(
        reading,
        clearance_m=config.clearance_m,
        hard_clearance_m=config.hard_clearance_m,
        target_height_m=config.target_height_m,
        max_speed_m_s=config.avoidance_speed_m_s,
        range_rate_m_s=range_rate_m_s,
        ttc_horizon_s=config.ttc_horizon_s,
        ttc_hard_s=config.ttc_hard_s,
        ttc_gain=config.ttc_gain,
    )
    pressure = max(
        target_path_pressure(
            reading,
            config.direction_deg,
            clearance_m=config.clearance_m,
            hard_clearance_m=config.hard_clearance_m,
            range_rate_m_s=range_rate_m_s,
            ttc_horizon_s=config.ttc_horizon_s,
            ttc_hard_s=config.ttc_hard_s,
        ),
        keepout_pressure(
        min_horizontal_range_m(reading),
        clearance_m=config.clearance_m,
        hard_clearance_m=config.hard_clearance_m,
        ),
    )
    cruise_scale = float(np.clip(1.0 - config.slowdown_gain * pressure, 0.0, 1.0))
    command = AvoidanceCommand(
        vx_m_s=cruise_scale * cruise[0] + config.avoidance_gain * avoidance.vx_m_s,
        vy_m_s=cruise_scale * cruise[1] + config.avoidance_gain * avoidance.vy_m_s,
        yawrate_deg_s=0.0,
        zdistance_m=config.target_height_m,
    )
    return clip_horizontal_norm(command, max_speed_m_s=config.max_speed_m_s)


def cruise_vector(direction_deg: float, speed_m_s: float) -> tuple[float, float]:
    angle = radians(direction_deg)
    return speed_m_s * cos(angle), speed_m_s * sin(angle)


def keepout_pressure(distance_m: float, *, clearance_m: float, hard_clearance_m: float) -> float:
    if clearance_m <= hard_clearance_m:
        raise ValueError("clearance_m must be greater than hard_clearance_m")
    scaled = (clearance_m - distance_m) / (clearance_m - hard_clearance_m)
    return float(np.sqrt(np.clip(scaled, 0.0, 1.0)))


def target_path_pressure(
    reading: RangerReading,
    direction_deg: float,
    *,
    clearance_m: float,
    hard_clearance_m: float,
    range_rate_m_s: RangerReading | None = None,
    ttc_horizon_s: float = 0.0,
    ttc_hard_s: float = 0.12,
) -> float:
    vx, vy = cruise_vector(direction_deg, 1.0)
    pressures = []
    if vx > 1e-6:
        pressures.append(_directional_pressure(reading.front_m, getattr(range_rate_m_s, "front_m", 0.0), clearance_m, hard_clearance_m, ttc_horizon_s, ttc_hard_s))
    elif vx < -1e-6:
        pressures.append(_directional_pressure(reading.back_m, getattr(range_rate_m_s, "back_m", 0.0), clearance_m, hard_clearance_m, ttc_horizon_s, ttc_hard_s))
    if vy > 1e-6:
        pressures.append(_directional_pressure(reading.left_m, getattr(range_rate_m_s, "left_m", 0.0), clearance_m, hard_clearance_m, ttc_horizon_s, ttc_hard_s))
    elif vy < -1e-6:
        pressures.append(_directional_pressure(reading.right_m, getattr(range_rate_m_s, "right_m", 0.0), clearance_m, hard_clearance_m, ttc_horizon_s, ttc_hard_s))
    return max(pressures, default=0.0)


def _directional_pressure(distance_m: float, rate_m_s: float, clearance_m: float, hard_clearance_m: float, ttc_horizon_s: float, ttc_hard_s: float) -> float:
    pressure = keepout_pressure(distance_m, clearance_m=clearance_m, hard_clearance_m=hard_clearance_m)
    if ttc_horizon_s > ttc_hard_s:
        ttc_s = time_to_collision_s(distance_m, rate_m_s)
        if np.isfinite(ttc_s) and ttc_s < ttc_horizon_s:
            pressure = max(pressure, float(np.sqrt(np.clip((ttc_horizon_s - ttc_s) / (ttc_horizon_s - ttc_hard_s), 0.0, 1.0))))
    return pressure


def clip_horizontal_norm(command: AvoidanceCommand, *, max_speed_m_s: float) -> AvoidanceCommand:
    vector = np.asarray([command.vx_m_s, command.vy_m_s], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= max_speed_m_s or norm <= 1e-9:
        return command
    scaled = vector * (max_speed_m_s / norm)
    return AvoidanceCommand(float(scaled[0]), float(scaled[1]), command.yawrate_deg_s, command.zdistance_m)
