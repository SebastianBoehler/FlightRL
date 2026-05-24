from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from .env import ACTION_DIM, OBSERVATION_DIM, SixDofCrazyflieEnv, quat_to_yaw, wrap_angle
from .geometry import quat_to_matrix
from .yaw import circle_tangent_yaw


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
        return self.net(torch.as_tensor(observations, dtype=torch.float32))


def teacher_actions(env: SixDofCrazyflieEnv, task: str = "position_yaw") -> np.ndarray:
    if task == "obstacle_avoidance":
        return obstacle_teacher(env)
    if task == "circle":
        return circle_teacher(env)
    if task == "attitude":
        return attitude_teacher(env)
    return position_yaw_teacher(env)


def position_yaw_teacher(env: SixDofCrazyflieEnv) -> np.ndarray:
    error = env.target_position - env.position
    desired_acc = 3.0 * error - 1.4 * env.velocity
    desired_yaw_rate = 1.8 * wrap_angle(env.target_yaw - quat_to_yaw(env.quaternion))
    return action_from_desired_acc(env, desired_acc, desired_yaw_rate)


def obstacle_teacher(env: SixDofCrazyflieEnv) -> np.ndarray:
    error = env.target_position - env.position
    desired_acc = 1.5 * error - 1.0 * env.velocity
    clearance = 0.55
    body_push = np.zeros_like(env.position)
    body_push[:, 0] += np.maximum(0.0, clearance - env.ranges_m[:, 1])
    body_push[:, 0] -= np.maximum(0.0, clearance - env.ranges_m[:, 0])
    body_push[:, 1] += np.maximum(0.0, clearance - env.ranges_m[:, 3])
    body_push[:, 1] -= np.maximum(0.0, clearance - env.ranges_m[:, 2])
    rotation = quat_to_matrix(env.quaternion)
    desired_acc += np.einsum("nij,nj->ni", rotation, 5.0 * body_push, optimize=True)
    yaw_rate = 1.2 * wrap_angle(env.target_yaw - quat_to_yaw(env.quaternion))
    return action_from_desired_acc(env, desired_acc, yaw_rate)


def attitude_teacher(env: SixDofCrazyflieEnv) -> np.ndarray:
    yaw = quat_to_yaw(env.quaternion)
    target_roll = 0.35 * np.tanh(env.target_position[:, 1])
    target_pitch = 0.35 * np.tanh(env.target_position[:, 0])
    roll, pitch = roll_pitch_from_quat(env.quaternion)
    rates = np.stack(
        [
            3.0 * (target_roll - roll),
            3.0 * (target_pitch - pitch),
            1.6 * wrap_angle(env.target_yaw - yaw),
        ],
        axis=1,
    )
    thrust = 1.0 / np.maximum(quat_to_matrix(env.quaternion)[:, 2, 2], 0.35)
    vertical = 1.5 * (env.target_position[:, 2] - env.position[:, 2]) - 0.5 * env.velocity[:, 2]
    action = np.zeros((env.num_envs, ACTION_DIM), dtype=np.float32)
    action[:, 0] = np.clip((thrust + 0.12 * vertical - 1.0) / 0.75, -1.0, 1.0)
    action[:, 1:4] = np.clip(rates / env.max_rate, -1.0, 1.0)
    return action


def circle_teacher(env: SixDofCrazyflieEnv) -> np.ndarray:
    center = env.target_position.copy()
    center[:, 2] = 0.65
    radial = env.position - center
    radial[:, 2] = 0.0
    radius = np.maximum(np.linalg.norm(radial[:, :2], axis=1, keepdims=True), 0.2)
    tangent = np.concatenate([-radial[:, 1:2], radial[:, 0:1], np.zeros((env.num_envs, 1), dtype=np.float32)], axis=1)
    tangent /= radius
    desired_velocity = 0.45 * tangent - 0.6 * (radius - 0.75) * radial / radius
    desired_velocity[:, 2] = 0.7 * (0.65 - env.position[:, 2])
    desired_acc = 2.0 * (desired_velocity - env.velocity)
    yaw_rate = 1.8 * wrap_angle(circle_tangent_yaw(env) - quat_to_yaw(env.quaternion))
    return action_from_desired_acc(env, desired_acc, yaw_rate)


def action_from_desired_acc(env: SixDofCrazyflieEnv, desired_acc: np.ndarray, yaw_rate: np.ndarray) -> np.ndarray:
    yaw = quat_to_yaw(env.quaternion)
    cy, sy = np.cos(yaw), np.sin(yaw)
    ax_body = cy * desired_acc[:, 0] + sy * desired_acc[:, 1]
    ay_body = -sy * desired_acc[:, 0] + cy * desired_acc[:, 1]
    roll, pitch = roll_pitch_from_quat(env.quaternion)
    roll_cmd = np.clip(-ay_body / env.gravity, -0.55, 0.55)
    pitch_cmd = np.clip(ax_body / env.gravity, -0.55, 0.55)
    thrust = env.gravity + desired_acc[:, 2]
    action = np.zeros((env.num_envs, ACTION_DIM), dtype=np.float32)
    action[:, 0] = np.clip((thrust / env.gravity - 1.0) / 0.75, -1.0, 1.0)
    action[:, 1] = np.clip(3.4 * (roll_cmd - roll) - 0.25 * env.body_rates[:, 0], -env.max_rate[0], env.max_rate[0]) / env.max_rate[0]
    action[:, 2] = np.clip(3.4 * (pitch_cmd - pitch) - 0.25 * env.body_rates[:, 1], -env.max_rate[1], env.max_rate[1]) / env.max_rate[1]
    action[:, 3] = np.clip(yaw_rate - 0.2 * env.body_rates[:, 2], -env.max_rate[2], env.max_rate[2]) / env.max_rate[2]
    return action


def roll_pitch_from_quat(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = quaternions
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    return roll.astype(np.float32), pitch.astype(np.float32)
