from __future__ import annotations

import numpy as np

from .geometry import quat_to_matrix
from .rl import position_error, rollout_reward


def drift_recovery_reward(env, done: np.ndarray, previous_error: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return _drift_recovery_reward(env, done, previous_error, actions, aggressive=False)


def aggressive_drift_recovery_reward(env, done: np.ndarray, previous_error: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return _drift_recovery_reward(env, done, previous_error, actions, aggressive=True)


def hover_transfer_reward(env, base_reward: np.ndarray, done: np.ndarray, previous_error: np.ndarray, actions: np.ndarray, mode: str, tasks: tuple[str, ...], task_indices: np.ndarray) -> np.ndarray:
    reward = rollout_reward(env, base_reward, done, previous_error, actions, "live_stable_clearance", tasks=tasks, task_indices=task_indices)
    min_clearance = np.min(env.ranges_m[:, :4], axis=1)
    open_space = np.clip((min_clearance - 0.45) / 0.20, 0.0, 1.0)
    horizontal_speed = np.linalg.norm(env.velocity[:, :2], axis=1)
    tilt_action = np.linalg.norm(actions[:, 1:3], axis=1)
    if mode == "puffer_hover_transfer":
        penalty = 0.65 * horizontal_speed + 0.15 * tilt_action
    else:
        excess_speed = np.maximum(0.0, horizontal_speed - 0.45)
        penalty = 1.20 * horizontal_speed + 2.00 * excess_speed + 0.25 * tilt_action
    return (reward - open_space * penalty).astype(np.float32)


def precontact_transfer_reward(
    env,
    done: np.ndarray,
    previous_error: np.ndarray,
    actions: np.ndarray,
    *,
    previous_action: np.ndarray | None = None,
) -> np.ndarray:
    current_error = position_error(env)
    progress = previous_error - current_error
    min_clearance = np.min(env.ranges_m[:, :4], axis=1)
    open_space = np.clip((min_clearance - 0.45) / 0.25, 0.0, 1.0)
    horizontal_speed = np.linalg.norm(env.velocity[:, :2], axis=1)
    speed_excess = np.maximum(0.0, horizontal_speed - 0.35)
    vertical_speed = np.abs(env.velocity[:, 2])
    height_error = np.abs(env.target_position[:, 2] - env.position[:, 2])
    roll, pitch = roll_pitch_from_quat(env.quaternion)
    tilt = np.maximum(np.abs(roll), np.abs(pitch))
    tilt_excess = np.maximum(0.0, tilt - 0.45)
    tilt_action = np.linalg.norm(actions[:, 1:3], axis=1)
    action_delta = (
        np.zeros(env.num_envs, dtype=np.float32)
        if previous_action is None
        else np.linalg.norm(actions - previous_action, axis=1)
    )
    rate_pressure = np.maximum(0.0, tilt_action - 0.35)
    clearance_penalty = np.maximum(0.0, 0.35 - min_clearance)
    return (
        0.6
        + 7.0 * progress
        - 1.45 * current_error
        - open_space
        * (
            2.4 * horizontal_speed
            + 3.4 * speed_excess * speed_excess
            + 2.2 * tilt_excess * tilt_excess
            + 0.18 * tilt_action
            + 0.35 * action_delta
            + 0.65 * rate_pressure * rate_pressure
        )
        - 0.35 * vertical_speed
        - 0.50 * height_error
        - 4.5 * clearance_penalty
        - 0.02 * (np.abs(actions[:, 0]) + 0.5 * np.abs(actions[:, 3]))
        - 3.0 * done.astype(np.float32)
    ).astype(np.float32)


def precontact_drift_brake_reward(
    env,
    done: np.ndarray,
    previous_error: np.ndarray,
    actions: np.ndarray,
    *,
    previous_action: np.ndarray | None = None,
) -> np.ndarray:
    current_error = position_error(env)
    progress = previous_error - current_error
    min_clearance = np.min(env.ranges_m[:, :4], axis=1)
    open_space = np.clip((min_clearance - 0.50) / 0.25, 0.0, 1.0)
    horizontal_speed = np.linalg.norm(env.velocity[:, :2], axis=1)
    speed_excess = np.maximum(0.0, horizontal_speed - 0.25)
    body_velocity = np.einsum("nij,ni->nj", quat_to_matrix(env.quaternion), env.velocity, optimize=True)
    body_horizontal = body_velocity[:, :2]
    body_speed = np.linalg.norm(body_horizontal, axis=1)
    brake_direction = -body_horizontal / np.maximum(body_speed[:, None], 0.05)
    horizontal_control = np.stack([actions[:, 2], -actions[:, 1]], axis=1)
    brake_alignment = np.sum(horizontal_control * brake_direction, axis=1)
    wrong_way = np.maximum(0.0, -brake_alignment)
    weak_brake = np.maximum(0.0, np.minimum(body_speed, 1.5) * 0.30 - brake_alignment)
    vertical_speed = np.abs(env.velocity[:, 2])
    height_error = np.abs(env.target_position[:, 2] - env.position[:, 2])
    roll, pitch = roll_pitch_from_quat(env.quaternion)
    tilt = np.maximum(np.abs(roll), np.abs(pitch))
    tilt_excess = np.maximum(0.0, tilt - 0.42)
    tilt_action = np.linalg.norm(actions[:, 1:3], axis=1)
    action_delta = (
        np.zeros(env.num_envs, dtype=np.float32)
        if previous_action is None
        else np.linalg.norm(actions - previous_action, axis=1)
    )
    clearance_penalty = np.maximum(0.0, 0.35 - min_clearance)
    brake_bonus = np.maximum(0.0, brake_alignment) * np.minimum(body_speed, 1.5)
    return (
        0.55
        + 5.5 * progress
        - 1.25 * current_error
        + open_space * 0.55 * brake_bonus
        - open_space
        * (
            3.4 * horizontal_speed
            + 5.8 * speed_excess * speed_excess
            + 1.8 * wrong_way * wrong_way
            + 0.9 * weak_brake * weak_brake
            + 1.7 * tilt_excess * tilt_excess
            + 0.10 * tilt_action
            + 0.22 * action_delta
        )
        - 0.35 * vertical_speed
        - 0.45 * height_error
        - 4.5 * clearance_penalty
        - 0.02 * (np.abs(actions[:, 0]) + 0.5 * np.abs(actions[:, 3]))
        - 3.0 * done.astype(np.float32)
    ).astype(np.float32)


def startup_drift_recovery_reward(
    env,
    done: np.ndarray,
    previous_error: np.ndarray,
    actions: np.ndarray,
    *,
    previous_action: np.ndarray | None = None,
    previous_horizontal_speed: np.ndarray | None = None,
) -> np.ndarray:
    current_error = position_error(env)
    progress = previous_error - current_error
    min_clearance = np.min(env.ranges_m[:, :4], axis=1)
    open_space = np.clip((min_clearance - 0.50) / 0.25, 0.0, 1.0)
    horizontal_speed = np.linalg.norm(env.velocity[:, :2], axis=1)
    previous_speed = horizontal_speed if previous_horizontal_speed is None else previous_horizontal_speed
    speed_reduction = previous_speed - horizontal_speed
    speed_pressure = np.maximum(previous_speed, horizontal_speed)
    recovery_gate = open_space * np.clip((speed_pressure - 0.35) / 0.75, 0.0, 1.0)
    body_velocity = np.einsum("nij,ni->nj", quat_to_matrix(env.quaternion), env.velocity, optimize=True)
    body_horizontal = body_velocity[:, :2]
    body_speed = np.linalg.norm(body_horizontal, axis=1)
    brake_direction = -body_horizontal / np.maximum(body_speed[:, None], 0.05)
    horizontal_control = np.stack([actions[:, 2], -actions[:, 1]], axis=1)
    brake_alignment = np.sum(horizontal_control * brake_direction, axis=1)
    wrong_way = np.maximum(0.0, -brake_alignment)
    weak_brake = np.maximum(0.0, np.minimum(speed_pressure, 1.8) * 0.38 - brake_alignment)
    roll, pitch = roll_pitch_from_quat(env.quaternion)
    tilt_excess = np.maximum(0.0, np.maximum(np.abs(roll), np.abs(pitch)) - 0.42)
    tilt_action = np.linalg.norm(actions[:, 1:3], axis=1)
    action_delta = (
        np.zeros(env.num_envs, dtype=np.float32)
        if previous_action is None
        else np.linalg.norm(actions - previous_action, axis=1)
    )
    vertical_speed = np.abs(env.velocity[:, 2])
    height_error = np.abs(env.target_position[:, 2] - env.position[:, 2])
    clearance_penalty = np.maximum(0.0, 0.35 - min_clearance)
    target_weight = 1.0 - 0.85 * recovery_gate
    return (
        0.55
        + 3.0 * target_weight * progress
        - 1.15 * target_weight * current_error
        + 3.0 * recovery_gate * speed_reduction
        + 0.20 * open_space * np.maximum(0.0, brake_alignment) * np.minimum(speed_pressure, 1.8)
        - open_space
        * (
            5.8 * horizontal_speed
            + 12.0 * np.maximum(0.0, horizontal_speed - 0.25) ** 2
            + 2.4 * recovery_gate * wrong_way * wrong_way
            + 1.6 * recovery_gate * weak_brake * weak_brake
            + 2.2 * tilt_excess * tilt_excess
            + 0.10 * tilt_action
            + 0.20 * action_delta
        )
        - 0.35 * vertical_speed
        - 0.45 * height_error
        - 4.5 * clearance_penalty
        - 0.02 * (np.abs(actions[:, 0]) + 0.5 * np.abs(actions[:, 3]))
        - 3.0 * done.astype(np.float32)
    ).astype(np.float32)


def _drift_recovery_reward(env, done: np.ndarray, previous_error: np.ndarray, actions: np.ndarray, *, aggressive: bool) -> np.ndarray:
    current_error = position_error(env)
    progress = previous_error - current_error
    min_clearance = np.min(env.ranges_m[:, :4], axis=1)
    open_space = np.clip((min_clearance - 0.45) / 0.20, 0.0, 1.0)
    horizontal_speed = np.linalg.norm(env.velocity[:, :2], axis=1)
    vertical_speed = np.abs(env.velocity[:, 2])
    height_error = np.abs(env.target_position[:, 2] - env.position[:, 2])
    control = np.linalg.norm(actions, axis=1)
    tilt_action = np.linalg.norm(actions[:, 1:3], axis=1)
    clearance_penalty = np.maximum(0.0, 0.35 - min_clearance)
    speed_excess = np.maximum(0.0, horizontal_speed - 0.35)
    if aggressive:
        drift_pressure = np.linalg.norm(env.velocity[:, :2] + 0.75 * (env.position[:, :2] - env.target_position[:, :2]), axis=1)
        tilt_weight = np.where(drift_pressure > 0.45, 0.035, 0.16)
        return (
            0.5
            + 6.5 * progress
            - 1.55 * current_error
            - open_space * (2.8 * horizontal_speed + 4.2 * speed_excess * speed_excess + tilt_weight * tilt_action)
            - 0.35 * vertical_speed
            - 0.45 * height_error
            - 4.0 * clearance_penalty
            - 0.025 * (np.abs(actions[:, 0]) + 0.5 * np.abs(actions[:, 3]))
            - 3.0 * done.astype(np.float32)
        ).astype(np.float32)
    reward = (
        0.5
        + 5.0 * progress
        - 1.25 * current_error
        - open_space * (1.6 * horizontal_speed + 2.2 * speed_excess + 0.20 * tilt_action)
        - 0.45 * vertical_speed
        - 0.55 * height_error
        - 4.0 * clearance_penalty
        - 0.02 * control
        - 3.0 * done.astype(np.float32)
    )
    return reward.astype(np.float32)


def roll_pitch_from_quat(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = quaternions
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    return roll.astype(np.float32), pitch.astype(np.float32)
