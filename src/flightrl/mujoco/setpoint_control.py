from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flightrl.sixdof.env import quat_to_yaw
from flightrl.sixdof.policies import roll_pitch_from_quat


@dataclass(frozen=True, slots=True)
class VisualSetpointConfig:
    max_horizontal_speed_m_s: float = 0.20
    max_vertical_speed_m_s: float = 0.10
    max_yawrate_deg_s: float = 60.0
    physics_substeps: int = 2
    velocity_gain: float = 3.0
    attitude_gain: float = 6.0
    vertical_gain: float = 2.0
    success_radius_m: float = 0.16


def firmware_setpoint_actions(sim, commands: np.ndarray, control: VisualSetpointConfig) -> np.ndarray:
    target_body_velocity = commands[:, :3] * np.asarray(
        [
            control.max_horizontal_speed_m_s,
            control.max_horizontal_speed_m_s,
            control.max_vertical_speed_m_s,
        ],
        dtype=np.float32,
    )
    yaw = quat_to_yaw(sim.quaternion)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    current_body_velocity = np.column_stack(
        (
            cosine * sim.velocity[:, 0] + sine * sim.velocity[:, 1],
            -sine * sim.velocity[:, 0] + cosine * sim.velocity[:, 1],
        )
    )
    velocity_error_body = target_body_velocity[:, :2] - current_body_velocity
    desired_pitch = np.clip(
        control.velocity_gain * velocity_error_body[:, 0] / sim.gravity,
        -0.25,
        0.25,
    )
    desired_roll = np.clip(
        -control.velocity_gain * velocity_error_body[:, 1] / sim.gravity,
        -0.25,
        0.25,
    )
    roll, pitch = roll_pitch_from_quat(sim.quaternion)
    roll_rate = control.attitude_gain * (desired_roll - roll)
    pitch_rate = control.attitude_gain * (desired_pitch - pitch)
    thrust = control.vertical_gain * (
        target_body_velocity[:, 2] - sim.velocity[:, 2]
    ) / max(sim.gravity, 1e-6)
    target_yaw_rate = commands[:, 3] * np.deg2rad(control.max_yawrate_deg_s)
    return np.column_stack(
        (
            np.clip(thrust, -1.0, 1.0),
            np.clip(roll_rate / sim.max_rate[0], -1.0, 1.0),
            np.clip(pitch_rate / sim.max_rate[1], -1.0, 1.0),
            np.clip(target_yaw_rate / sim.max_rate[2], -1.0, 1.0),
        )
    ).astype(np.float32)
