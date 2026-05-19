from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from .avoidance_policy import RangerReading, reactive_clearance_command, vertical_velocity_from_height_error


HOLD_OBSERVATION_DIM = 24
HOLD_OUTPUT_SCALE = np.asarray([0.35, 0.35, 0.22, 60.0], dtype=np.float32)
HOLD_LOG_VARIABLES = (
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.vz",
    "gyro.x",
    "gyro.y",
    "gyro.z",
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "range.zrange",
    "pm.vbat",
)


@dataclass(frozen=True, slots=True)
class HoldState:
    ranges: RangerReading
    x_m: float
    y_m: float
    z_m: float
    vx_m_s: float
    vy_m_s: float
    vz_m_s: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    gyro_x_rad_s: float
    gyro_y_rad_s: float
    gyro_z_rad_s: float
    target_x_m: float
    target_y_m: float
    target_z_m: float


@dataclass(frozen=True, slots=True)
class HoldCommand:
    vx_m_s: float
    vy_m_s: float
    vz_m_s: float
    yawrate_deg_s: float

    def clipped(self, *, max_speed: float = 0.35, max_vertical_speed: float = 0.22, max_yawrate: float = 60.0) -> "HoldCommand":
        return HoldCommand(
            vx_m_s=float(np.clip(self.vx_m_s, -max_speed, max_speed)),
            vy_m_s=float(np.clip(self.vy_m_s, -max_speed, max_speed)),
            vz_m_s=float(np.clip(self.vz_m_s, -max_vertical_speed, max_vertical_speed)),
            yawrate_deg_s=float(np.clip(self.yawrate_deg_s, -max_yawrate, max_yawrate)),
        )


class RangerHoldPolicy(nn.Module):
    def __init__(self, hidden_size: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HOLD_OBSERVATION_DIM, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 4),
        )

    def forward(self, observations):
        tensor = torch.as_tensor(observations, dtype=torch.float32)
        return self.net(tensor)


def normalize_hold_state(state: HoldState) -> np.ndarray:
    ranges = state.ranges
    values = [
        _range_norm(ranges.front_m),
        _range_norm(ranges.back_m),
        _range_norm(ranges.left_m),
        _range_norm(ranges.right_m),
        _range_norm(ranges.up_m),
        _range_norm(ranges.zrange_m),
        _clip_div(state.target_x_m - state.x_m, 2.0),
        _clip_div(state.target_y_m - state.y_m, 2.0),
        _clip_div(state.target_z_m - state.z_m, 1.0),
        _clip_div(state.vx_m_s, 1.5),
        _clip_div(state.vy_m_s, 1.5),
        _clip_div(state.vz_m_s, 1.0),
        sin(state.roll_rad),
        cos(state.roll_rad),
        sin(state.pitch_rad),
        cos(state.pitch_rad),
        sin(state.yaw_rad),
        cos(state.yaw_rad),
        _clip_div(state.gyro_x_rad_s, 8.0),
        _clip_div(state.gyro_y_rad_s, 8.0),
        _clip_div(state.gyro_z_rad_s, 8.0),
        _clip_div(state.target_x_m, 2.0),
        _clip_div(state.target_y_m, 2.0),
        _clip_div(state.target_z_m, 1.0),
    ]
    return np.asarray(values, dtype=np.float32)


def teacher_hold_command(
    state: HoldState,
    *,
    clearance_m: float = 0.45,
    hard_clearance_m: float = 0.10,
    max_speed_m_s: float = 0.35,
    max_vertical_speed_m_s: float = 0.22,
) -> HoldCommand:
    avoidance = reactive_clearance_command(
        state.ranges,
        clearance_m=clearance_m,
        hard_clearance_m=hard_clearance_m,
        target_height_m=state.target_z_m,
        max_speed_m_s=max_speed_m_s,
    )
    hold_vx = 0.75 * (state.target_x_m - state.x_m) - 0.35 * state.vx_m_s - 0.10 * state.pitch_rad
    hold_vy = 0.75 * (state.target_y_m - state.y_m) - 0.35 * state.vy_m_s + 0.10 * state.roll_rad
    hold_vz = 1.10 * (state.target_z_m - state.z_m) - 0.40 * state.vz_m_s
    range_vz = vertical_velocity_from_height_error(
        avoidance,
        state.ranges,
        gain=0.9,
        max_vertical_speed_m_s=max_vertical_speed_m_s,
    )
    yawrate = -35.0 * state.yaw_rad - 6.0 * state.gyro_z_rad_s
    command = HoldCommand(
        vx_m_s=hold_vx + avoidance.vx_m_s,
        vy_m_s=hold_vy + avoidance.vy_m_s,
        vz_m_s=hold_vz + range_vz,
        yawrate_deg_s=yawrate,
    )
    return command.clipped(max_speed=max_speed_m_s, max_vertical_speed=max_vertical_speed_m_s)


def command_from_hold_model(model: RangerHoldPolicy, state: HoldState) -> HoldCommand:
    obs = normalize_hold_state(state)[None, :]
    with torch.no_grad():
        raw = model(obs).squeeze(0).cpu().numpy() * HOLD_OUTPUT_SCALE
    return HoldCommand(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])).clipped()


def hold_command_array(command: HoldCommand) -> np.ndarray:
    return np.asarray([command.vx_m_s, command.vy_m_s, command.vz_m_s, command.yawrate_deg_s], dtype=np.float32)


def normalized_hold_command_array(command: HoldCommand) -> np.ndarray:
    return hold_command_array(command) / HOLD_OUTPUT_SCALE


def hold_command_row(command: HoldCommand) -> dict[str, float]:
    return {
        "vx_m_s": command.vx_m_s,
        "vy_m_s": command.vy_m_s,
        "vz_m_s": command.vz_m_s,
        "yawrate_deg_s": command.yawrate_deg_s,
    }


def hold_state_from_telemetry(values: Mapping[str, float], target: tuple[float, float, float]) -> HoldState:
    return HoldState(
        ranges=RangerReading(
            front_m=_range_m(values, "range.front"),
            back_m=_range_m(values, "range.back"),
            left_m=_range_m(values, "range.left"),
            right_m=_range_m(values, "range.right"),
            up_m=_range_m(values, "range.up"),
            zrange_m=_range_m(values, "range.zrange"),
        ),
        x_m=_get(values, "stateEstimate.x"),
        y_m=_get(values, "stateEstimate.y"),
        z_m=_get(values, "stateEstimate.z"),
        vx_m_s=_get(values, "stateEstimate.vx"),
        vy_m_s=_get(values, "stateEstimate.vy"),
        vz_m_s=_get(values, "stateEstimate.vz"),
        roll_rad=_deg_to_rad(_get(values, "stabilizer.roll")),
        pitch_rad=_deg_to_rad(_get(values, "stabilizer.pitch")),
        yaw_rad=_deg_to_rad(_get(values, "stabilizer.yaw")),
        gyro_x_rad_s=_deg_to_rad(_get(values, "gyro.x")),
        gyro_y_rad_s=_deg_to_rad(_get(values, "gyro.y")),
        gyro_z_rad_s=_deg_to_rad(_get(values, "gyro.z")),
        target_x_m=target[0],
        target_y_m=target[1],
        target_z_m=target[2],
    )


def sample_hold_states(count: int, rng: np.random.Generator) -> list[HoldState]:
    states = []
    for _ in range(count):
        target = rng.uniform([-0.8, -0.8, 0.35], [0.8, 0.8, 0.75])
        position = target + rng.normal(0.0, [0.55, 0.55, 0.18])
        ranges = rng.uniform(0.12, 3.2, size=6)
        if rng.random() < 0.55:
            ranges[rng.integers(0, 6)] = rng.uniform(0.07, 0.45)
        ranges[5] = max(0.08, position[2] + rng.normal(0.0, 0.05))
        states.append(
            HoldState(
                ranges=RangerReading(*[float(v) for v in ranges]),
                x_m=float(position[0]),
                y_m=float(position[1]),
                z_m=float(position[2]),
                vx_m_s=float(rng.normal(0.0, 0.45)),
                vy_m_s=float(rng.normal(0.0, 0.45)),
                vz_m_s=float(rng.normal(0.0, 0.25)),
                roll_rad=float(rng.uniform(-0.9, 0.9)),
                pitch_rad=float(rng.uniform(-0.9, 0.9)),
                yaw_rad=float(rng.uniform(-pi, pi)),
                gyro_x_rad_s=float(rng.normal(0.0, 2.5)),
                gyro_y_rad_s=float(rng.normal(0.0, 2.5)),
                gyro_z_rad_s=float(rng.normal(0.0, 2.5)),
                target_x_m=float(target[0]),
                target_y_m=float(target[1]),
                target_z_m=float(target[2]),
            )
        )
    return states


def _range_norm(value_m: float) -> float:
    return float(np.clip(value_m / 4.0, 0.0, 1.0))


def _clip_div(value: float, scale: float) -> float:
    return float(np.clip(value / scale, -1.0, 1.0))


def _get(values: Mapping[str, float], key: str) -> float:
    try:
        return float(values.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _deg_to_rad(value: float) -> float:
    return float(value * pi / 180.0)


def _range_m(values: Mapping[str, float], key: str) -> float:
    raw = _get(values, key)
    if raw >= 32000.0 or raw <= 0.0:
        return 4.0
    return raw / 1000.0
