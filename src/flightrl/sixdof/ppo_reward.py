from __future__ import annotations

import numpy as np

from .circle import circle_orbit_error_from_arrays
from .env import quat_to_yaw, wrap_angle
from .geometry import quat_to_matrix
from .yaw import yaw_error_for_task_indices


def position_error(env) -> np.ndarray:
    return np.linalg.norm(
        env.target_position - env.position,
        axis=1,
    ).astype(np.float32)


def position_error_for_task_indices(
    env,
    tasks: tuple[str, ...],
    task_indices: np.ndarray,
) -> np.ndarray:
    errors = np.zeros(env.num_envs, dtype=np.float32)
    default_error = position_error(env)
    for index, task in enumerate(tasks):
        mask = task_indices == index
        if not np.any(mask):
            continue
        errors[mask] = (
            circle_position_error(env)[mask]
            if task == "circle"
            else default_error[mask]
        )
    return errors


def circle_position_error(
    env,
    target_radius_m: float = 0.75,
) -> np.ndarray:
    return circle_orbit_error_from_arrays(
        env.position,
        env.target_position,
        target_radius_m,
    )


def rollout_reward(
    env,
    base_reward: np.ndarray,
    done: np.ndarray,
    previous_error: np.ndarray,
    actions: np.ndarray,
    mode: str,
    *,
    tasks: tuple[str, ...] | None = None,
    task_indices: np.ndarray | None = None,
) -> np.ndarray:
    if mode == "env":
        return base_reward.copy()
    settings = {
        "progress": (0.25, 1.0, 0.1),
        "progress_clearance": (0.45, 2.5, 0.1),
        "progress_yaw_clearance": (0.45, 2.5, 0.6),
    }
    if mode in settings:
        threshold, clearance_weight, yaw_weight = settings[mode]
        return shaped_progress_reward(
            env,
            done,
            previous_error,
            actions,
            clearance_threshold=threshold,
            clearance_weight=clearance_weight,
            yaw_weight=yaw_weight,
            tasks=tasks,
            task_indices=task_indices,
        )
    if mode == "live_clearance":
        return shaped_progress_reward(
            env,
            done,
            previous_error,
            actions,
            clearance_threshold=0.65,
            clearance_weight=5.0,
            yaw_weight=0.1,
            clearance_bonus_weight=0.15,
            tasks=tasks,
            task_indices=task_indices,
        )
    if mode == "live_stable_clearance":
        return shaped_progress_reward(
            env,
            done,
            previous_error,
            actions,
            clearance_threshold=0.65,
            clearance_weight=5.0,
            yaw_weight=0.1,
            clearance_bonus_weight=0.15,
            speed_weight=0.12,
            open_space_speed_weight=0.35,
            open_space_action_weight=0.10,
            escape_velocity_weight=1.20,
            escape_action_weight=0.80,
            vertical_clearance_threshold=0.45,
            vertical_clearance_weight=3.00,
            vertical_escape_velocity_weight=0.50,
            vertical_escape_action_weight=0.25,
            terminal_penalty=2.0,
            tasks=tasks,
            task_indices=task_indices,
        )
    raise ValueError(f"unknown PPO reward mode {mode!r}")


def shaped_progress_reward(
    env,
    done: np.ndarray,
    previous_error: np.ndarray,
    actions: np.ndarray,
    *,
    clearance_threshold: float,
    clearance_weight: float,
    yaw_weight: float,
    tasks: tuple[str, ...] | None,
    task_indices: np.ndarray | None,
    clearance_bonus_weight: float = 0.0,
    speed_weight: float = 0.02,
    open_space_speed_weight: float = 0.0,
    open_space_action_weight: float = 0.0,
    escape_velocity_weight: float = 0.0,
    escape_action_weight: float = 0.0,
    vertical_clearance_threshold: float = 0.0,
    vertical_clearance_weight: float = 0.0,
    vertical_escape_velocity_weight: float = 0.0,
    vertical_escape_action_weight: float = 0.0,
    terminal_penalty: float = 1.0,
) -> np.ndarray:
    conditioned = tasks is not None and task_indices is not None
    current_error = (
        position_error_for_task_indices(env, tasks, task_indices)
        if conditioned
        else position_error(env)
    )
    progress = previous_error - current_error
    speed = np.linalg.norm(env.velocity, axis=1)
    yaw_error = (
        yaw_error_for_task_indices(env, tasks, task_indices)
        if conditioned
        else np.abs(
            wrap_angle(env.target_yaw - quat_to_yaw(env.quaternion))
        )
    )
    min_clearance = np.min(env.ranges_m[:, :4], axis=1)
    clearance_penalty = np.maximum(0.0, clearance_threshold - min_clearance)
    clearance_bonus = clearance_bonus_weight * np.minimum(
        min_clearance,
        clearance_threshold,
    )
    control = np.linalg.norm(actions, axis=1)
    open_space = np.clip((min_clearance - 0.45) / 0.20, 0.0, 1.0)
    horizontal_speed = np.linalg.norm(env.velocity[:, :2], axis=1)
    tilt_action = np.linalg.norm(actions[:, 1:3], axis=1)
    stable_penalty = open_space * (
        open_space_speed_weight * horizontal_speed
        + open_space_action_weight * tilt_action
    )
    escape_reward = clearance_escape_reward(
        env,
        actions,
        clearance_threshold,
        escape_velocity_weight,
        escape_action_weight,
    )
    vertical_reward = vertical_clearance_reward(
        env,
        actions,
        vertical_clearance_threshold,
        vertical_clearance_weight,
        vertical_escape_velocity_weight,
        vertical_escape_action_weight,
    )
    reward = (
        0.2
        + 3.0 * progress
        - 0.05 * current_error
        - speed_weight * speed
        - yaw_weight * yaw_error
        - clearance_weight * clearance_penalty
        + clearance_bonus
        - 0.01 * control
        - stable_penalty
        + escape_reward
        + vertical_reward
    )
    reward -= terminal_penalty * done.astype(np.float32)
    return reward.astype(np.float32)


def clearance_escape_reward(
    env,
    actions: np.ndarray,
    threshold: float,
    velocity_weight: float,
    action_weight: float,
) -> np.ndarray:
    if velocity_weight == 0.0 and action_weight == 0.0:
        return np.zeros(env.num_envs, dtype=np.float32)
    body_push_x = np.maximum(0.0, threshold - env.ranges_m[:, 1])
    body_push_x -= np.maximum(0.0, threshold - env.ranges_m[:, 0])
    body_push_y = np.maximum(0.0, threshold - env.ranges_m[:, 3])
    body_push_y -= np.maximum(0.0, threshold - env.ranges_m[:, 2])
    rotation = quat_to_matrix(env.quaternion)
    body_velocity = np.einsum(
        "nij,ni->nj",
        rotation,
        env.velocity,
        optimize=True,
    )
    velocity_alignment = (
        body_push_x * body_velocity[:, 0]
        + body_push_y * body_velocity[:, 1]
    )
    action_alignment = (
        body_push_x * actions[:, 2] - body_push_y * actions[:, 1]
    )
    return (
        velocity_weight * velocity_alignment
        + action_weight * action_alignment
    ).astype(np.float32)


def vertical_clearance_reward(
    env,
    actions: np.ndarray,
    threshold: float,
    clearance_weight: float,
    velocity_weight: float,
    action_weight: float,
) -> np.ndarray:
    if threshold <= 0.0:
        return np.zeros(env.num_envs, dtype=np.float32)
    top_pressure = np.maximum(0.0, threshold - env.ranges_m[:, 4])
    bottom_pressure = np.maximum(0.0, threshold - env.ranges_m[:, 5])
    vertical_push = bottom_pressure - top_pressure
    return (
        -clearance_weight * (top_pressure + bottom_pressure)
        + velocity_weight * vertical_push * env.velocity[:, 2]
        + action_weight * vertical_push * actions[:, 0]
    ).astype(np.float32)
