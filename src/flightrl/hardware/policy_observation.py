from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Mapping, Sequence

import numpy as np

from flightrl.config import FlightConfig


POLICY_LOG_VARIABLES = (
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.vz",
    "acc.x",
    "acc.y",
    "acc.z",
    "gyro.x",
    "gyro.y",
    "gyro.z",
    "pm.vbat",
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "range.zrange",
)


@dataclass(slots=True)
class PolicyObservationState:
    previous_action: np.ndarray


def initial_policy_state(config: FlightConfig) -> PolicyObservationState:
    return PolicyObservationState(previous_action=np.zeros(config.action_dim, dtype=np.float32))


def build_policy_observation(
    config: FlightConfig,
    telemetry: Mapping[str, float],
    state: PolicyObservationState,
    *,
    target: Sequence[float],
) -> np.ndarray:
    dyn = config.drone
    x = _get(telemetry, "stateEstimate.x")
    y = _get(telemetry, "stateEstimate.y")
    z = _get(telemetry, "stateEstimate.z")
    vx = _get(telemetry, "stateEstimate.vx")
    vy = _get(telemetry, "stateEstimate.vy")
    vz = _get(telemetry, "stateEstimate.vz")
    roll = _deg_to_rad(_get(telemetry, "stabilizer.roll"))
    pitch = _deg_to_rad(_get(telemetry, "stabilizer.pitch"))
    yaw = _deg_to_rad(_get(telemetry, "stabilizer.yaw"))
    max_rate = max(dyn.max_pitch_rate, 1e-6)
    target_x, target_y, target_z = target

    values = [
        *(_range_values(telemetry) if config.sensors.include_range_sensor else []),
        _safe_div(x, dyn.x_limit),
        _safe_div(y, dyn.x_limit),
        _safe_div(z, dyn.z_limit),
        _safe_div(vx, dyn.max_velocity),
        _safe_div(vy, dyn.max_velocity),
        _safe_div(vz, dyn.max_velocity),
        sin(roll),
        cos(roll),
        sin(pitch),
        cos(pitch),
        sin(yaw),
        cos(yaw),
        _safe_div(_deg_to_rad(_get(telemetry, "gyro.x")), max_rate),
        _safe_div(_deg_to_rad(_get(telemetry, "gyro.y")), max_rate),
        _safe_div(_deg_to_rad(_get(telemetry, "gyro.z")), max_rate),
        _get(telemetry, "acc.x"),
        _get(telemetry, "acc.y"),
        _get(telemetry, "acc.z"),
        _range_norm(telemetry, "range.front"),
        _range_norm(telemetry, "range.back"),
        _range_norm(telemetry, "range.left"),
        _range_norm(telemetry, "range.right"),
        _range_norm(telemetry, "range.up"),
        _range_norm(telemetry, "range.zrange"),
        _battery_norm(_get(telemetry, "pm.vbat")),
        _safe_div(target_x - x, dyn.x_limit),
        _safe_div(target_y - y, dyn.x_limit),
        _safe_div(target_z - z, dyn.z_limit),
        *state.previous_action.tolist(),
    ]
    return np.asarray(values, dtype=np.float32)


def update_previous_action(state: PolicyObservationState, action: np.ndarray) -> None:
    state.previous_action[:] = np.asarray(action, dtype=np.float32)


def _get(telemetry: Mapping[str, float], key: str) -> float:
    try:
        return float(telemetry.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _safe_div(value: float, scale: float) -> float:
    return 0.0 if abs(scale) < 1e-6 else float(value / scale)


def _deg_to_rad(value: float) -> float:
    return float(value * pi / 180.0)


def _range_norm(telemetry: Mapping[str, float], key: str) -> float:
    value_m = _get(telemetry, key) / 1000.0
    return float(np.clip(value_m / 4.0, 0.0, 1.0))


def _range_values(telemetry: Mapping[str, float]) -> list[float]:
    return [
        _range_norm(telemetry, "range.front"),
        _range_norm(telemetry, "range.back"),
        _range_norm(telemetry, "range.left"),
        _range_norm(telemetry, "range.right"),
        _range_norm(telemetry, "range.up"),
    ]


def _battery_norm(vbat: float) -> float:
    return float(np.clip((vbat - 3.3) / 0.9, 0.0, 1.0))
