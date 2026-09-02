from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from .env import ACTION_DIM, OBSERVATION_DIM, SixDofEnv, quat_to_yaw, wrap_angle
from .geometry import quat_to_matrix
from .tasks import TASKS
from .yaw import circle_tangent_yaw
from flightrl.vertical_clearance import vertical_clearance_push_np


TEACHER_PROFILES = ("default", "aggressive_open_stress", "open_space_stress", "bounded_recovery")


class SixDofPolicy(nn.Module):
    def __init__(self, hidden_size: int = 128, output_dim: int = ACTION_DIM, input_dim: int = OBSERVATION_DIM) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Tanh(),
        )

    def forward(self, observations):
        return torch.tanh(self.action_location(observations))

    def action_location(self, observations):
        hidden = torch.as_tensor(observations, dtype=torch.float32)
        for layer in tuple(self.net.children())[:-1]:
            hidden = layer(hidden)
        return hidden


def teacher_actions(env: SixDofEnv, task: str = "position_yaw") -> np.ndarray:
    if task == "obstacle_avoidance":
        return obstacle_teacher(env, profile=getattr(env, "teacher_profile", "default"))
    if task == "circle":
        return circle_teacher(env)
    if task == "position_yaw":
        return position_yaw_teacher(env)
    raise ValueError(f"unknown 6-DoF teacher task {task!r}; expected one of {TASKS}")


def position_yaw_teacher(env: SixDofEnv) -> np.ndarray:
    error = env.target_position - env.position
    desired_acc = 3.0 * error - 1.4 * env.velocity
    desired_yaw_rate = 1.8 * wrap_angle(env.target_yaw - quat_to_yaw(env.quaternion))
    return action_from_desired_acc(env, desired_acc, desired_yaw_rate)


def obstacle_teacher(env: SixDofEnv, *, profile: str = "default") -> np.ndarray:
    if profile not in TEACHER_PROFILES:
        raise ValueError(f"unknown teacher profile {profile!r}; expected one of {', '.join(TEACHER_PROFILES)}")
    if profile == "aggressive_open_stress":
        return obstacle_teacher_from_gains(env, position_gain=6.0, velocity_gain=8.0, push_gain=5.0, max_lean_rad=0.55, attitude_rate_gain=12.0)
    default = obstacle_teacher_from_gains(env, position_gain=1.5, velocity_gain=1.0, push_gain=5.0, max_lean_rad=0.55, attitude_rate_gain=3.4)
    if profile == "open_space_stress":
        aggressive = obstacle_teacher_from_gains(env, position_gain=6.0, velocity_gain=8.0, push_gain=5.0, max_lean_rad=0.55, attitude_rate_gain=12.0)
        return blend_open_space(default, aggressive, np.min(env.ranges_m[:, :4], axis=1), lower=0.55, upper=0.85)
    if profile == "bounded_recovery":
        recovery = obstacle_teacher_from_gains(env, position_gain=4.5, velocity_gain=8.0, push_gain=5.0, max_lean_rad=0.50, attitude_rate_gain=9.5)
        blended = blend_open_space(default, recovery, np.min(env.ranges_m[:, :4], axis=1), lower=0.55, upper=0.85)
        return clip_live_action_envelope(blended)
    return default


def blend_open_space(default: np.ndarray, open_space_action: np.ndarray, horizontal_clearance_m: np.ndarray, *, lower: float, upper: float) -> np.ndarray:
    weight = np.clip((horizontal_clearance_m - lower) / (upper - lower), 0.0, 1.0)
    weight = (weight * weight * (3.0 - 2.0 * weight)).astype(np.float32)[:, None]
    return ((1.0 - weight) * default + weight * open_space_action).astype(np.float32)


def clip_live_action_envelope(actions: np.ndarray) -> np.ndarray:
    clipped = actions.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], -0.25, 0.35)
    clipped[:, 1:3] = np.clip(clipped[:, 1:3], -0.64, 0.64)
    clipped[:, 3] = np.clip(clipped[:, 3], -0.35, 0.35)
    return clipped.astype(np.float32)


def obstacle_teacher_from_gains(
    env: SixDofEnv,
    *,
    position_gain: float,
    velocity_gain: float,
    push_gain: float,
    max_lean_rad: float,
    attitude_rate_gain: float,
) -> np.ndarray:
    error = env.target_position - env.position
    desired_acc = position_gain * error - velocity_gain * env.velocity
    horizontal_clearance = 0.55
    body_push = np.zeros_like(env.position)
    body_push[:, 0] += np.maximum(0.0, horizontal_clearance - env.ranges_m[:, 1])
    body_push[:, 0] -= np.maximum(0.0, horizontal_clearance - env.ranges_m[:, 0])
    body_push[:, 1] += np.maximum(0.0, horizontal_clearance - env.ranges_m[:, 3])
    body_push[:, 1] -= np.maximum(0.0, horizontal_clearance - env.ranges_m[:, 2])
    body_push[:, 2] += 0.30 * vertical_clearance_push_np(env.ranges_m[:, 4], env.ranges_m[:, 5])
    rotation = quat_to_matrix(env.quaternion)
    desired_acc += np.einsum("nij,nj->ni", rotation, push_gain * body_push, optimize=True)
    yaw_rate = 1.2 * wrap_angle(env.target_yaw - quat_to_yaw(env.quaternion))
    return action_from_desired_acc(env, desired_acc, yaw_rate, max_lean_rad=max_lean_rad, attitude_rate_gain=attitude_rate_gain)


def circle_teacher(env: SixDofEnv) -> np.ndarray:
    radial = env.position - env.target_position
    radial[:, 2] = 0.0
    radius = np.maximum(np.linalg.norm(radial[:, :2], axis=1, keepdims=True), 0.2)
    tangent = np.concatenate([-radial[:, 1:2], radial[:, 0:1], np.zeros((env.num_envs, 1), dtype=np.float32)], axis=1)
    tangent /= radius
    desired_velocity = 0.45 * tangent - 0.6 * (radius - 0.75) * radial / radius
    desired_velocity[:, 2] = 0.7 * (
        env.target_position[:, 2] - env.position[:, 2]
    )
    desired_acc = 2.0 * (desired_velocity - env.velocity)
    yaw_rate = 1.8 * wrap_angle(circle_tangent_yaw(env) - quat_to_yaw(env.quaternion))
    return action_from_desired_acc(env, desired_acc, yaw_rate)


def action_from_desired_acc(
    env: SixDofEnv,
    desired_acc: np.ndarray,
    yaw_rate: np.ndarray,
    *,
    max_lean_rad: float = 0.55,
    attitude_rate_gain: float = 3.4,
) -> np.ndarray:
    yaw = quat_to_yaw(env.quaternion)
    cy, sy = np.cos(yaw), np.sin(yaw)
    ax_body = cy * desired_acc[:, 0] + sy * desired_acc[:, 1]
    ay_body = -sy * desired_acc[:, 0] + cy * desired_acc[:, 1]
    roll, pitch = roll_pitch_from_quat(env.quaternion)
    roll_cmd = np.clip(-ay_body / env.gravity, -max_lean_rad, max_lean_rad)
    pitch_cmd = np.clip(ax_body / env.gravity, -max_lean_rad, max_lean_rad)
    thrust = env.gravity + desired_acc[:, 2]
    action = np.zeros((env.num_envs, ACTION_DIM), dtype=np.float32)
    action[:, 0] = np.clip((thrust / env.gravity - 1.0) / 0.75, -1.0, 1.0)
    action[:, 1] = np.clip(attitude_rate_gain * (roll_cmd - roll) - 0.25 * env.body_rates[:, 0], -env.max_rate[0], env.max_rate[0]) / env.max_rate[0]
    action[:, 2] = np.clip(attitude_rate_gain * (pitch_cmd - pitch) - 0.25 * env.body_rates[:, 1], -env.max_rate[1], env.max_rate[1]) / env.max_rate[1]
    action[:, 3] = np.clip(yaw_rate - 0.2 * env.body_rates[:, 2], -env.max_rate[2], env.max_rate[2]) / env.max_rate[2]
    return action


def roll_pitch_from_quat(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = quaternions
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    return roll.astype(np.float32), pitch.astype(np.float32)
