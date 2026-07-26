from __future__ import annotations

from math import radians

import numpy as np

from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.env import euler_to_quat


def live_env_from_telemetry(
    env: SixDofCrazyflieEnv,
    telemetry: dict[str, float],
    *,
    target: np.ndarray,
    target_yaw: float,
) -> None:
    env.position[0] = [
        value(telemetry, "stateEstimate.x"),
        value(telemetry, "stateEstimate.y"),
        value(telemetry, "stateEstimate.z"),
    ]
    env.velocity[0] = [
        value(telemetry, "stateEstimate.vx"),
        value(telemetry, "stateEstimate.vy"),
        value(telemetry, "stateEstimate.vz"),
    ]
    roll = radians(value(telemetry, "stabilizer.roll", fallback="stateEstimate.roll"))
    pitch = radians(value(telemetry, "stabilizer.pitch", fallback="stateEstimate.pitch"))
    yaw = radians(value(telemetry, "stabilizer.yaw", fallback="stateEstimate.yaw"))
    env.quaternion[0] = euler_to_quat(np.asarray([roll]), np.asarray([pitch]), np.asarray([yaw]))[0]
    env.body_rates[0] = [
        radians(value(telemetry, "gyro.x")),
        radians(value(telemetry, "gyro.y")),
        radians(value(telemetry, "gyro.z")),
    ]
    env.ranges_m[0] = [
        range_m(telemetry, "range.front"),
        range_m(telemetry, "range.back"),
        range_m(telemetry, "range.left"),
        range_m(telemetry, "range.right"),
        range_m(telemetry, "range.up"),
        range_m(telemetry, "range.zrange"),
    ]
    env.target_position[0] = target
    env.target_yaw[0] = target_yaw


def target_from_telemetry(telemetry: dict[str, float], fallback: np.ndarray) -> np.ndarray:
    target = np.asarray(fallback, dtype=np.float32).copy()
    for index, key in enumerate(("target_x", "target_y", "target_z")):
        if key in telemetry:
            target[index] = value(telemetry, key)
    return target


def value(telemetry: dict[str, float], key: str, *, fallback: str | None = None) -> float:
    raw = telemetry.get(key)
    if raw is None and fallback is not None:
        raw = telemetry.get(fallback)
    try:
        return float(raw if raw is not None else 0.0)
    except (TypeError, ValueError):
        return 0.0


def range_m(telemetry: dict[str, float], key: str) -> float:
    raw = value(telemetry, key, fallback=None)
    if raw <= 0.0 or not np.isfinite(raw):
        return 4.0
    return 4.0 if raw >= 32000.0 else raw / 1000.0


def action_columns(prefix: str, action: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_thrust": float(action[0]),
        f"{prefix}_roll_rate": float(action[1]),
        f"{prefix}_pitch_rate": float(action[2]),
        f"{prefix}_yaw_rate": float(action[3]),
    }
