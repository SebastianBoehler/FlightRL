from __future__ import annotations

import numpy as np

from .circle import (
    circle_orbit_error_from_arrays,
    circle_tangent_yaw_from_arrays,
)


CIRCLE_TASK_ID = 3


def task_target_yaw(
    position: np.ndarray,
    target_position: np.ndarray,
    target_yaw: np.ndarray,
    task_ids: np.ndarray,
) -> np.ndarray:
    circle_mask = task_ids == CIRCLE_TASK_ID
    if not np.any(circle_mask):
        return target_yaw
    effective = target_yaw.copy()
    tangent = circle_tangent_yaw_from_arrays(position, target_position)
    effective[circle_mask] = tangent[circle_mask]
    return effective


def write_yaw_observation(
    observations: np.ndarray,
    target_yaw: np.ndarray,
    yaw: np.ndarray,
) -> None:
    error = wrap_angle(target_yaw - yaw)
    observations[:, 16] = np.sin(error)
    observations[:, 17] = np.cos(error)


def default_task_reward(env, actions: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    circle_mask = env.native_task_ids == CIRCLE_TASK_ID
    position_error = np.linalg.norm(
        env.target_position - env.position,
        axis=1,
    )
    if np.any(circle_mask):
        orbit_error = circle_orbit_error_from_arrays(
            env.position,
            env.target_position,
        )
        position_error[circle_mask] = orbit_error[circle_mask]
    speed = np.linalg.norm(env.velocity, axis=1)
    target_yaw = task_target_yaw(
        env.position,
        env.target_position,
        env.target_yaw,
        env.native_task_ids,
    )
    yaw_error = np.abs(wrap_angle(target_yaw - yaw))
    clearance_penalty = np.maximum(
        0.0,
        0.35 - np.min(env.ranges_m[:, :4], axis=1),
    )
    control = np.linalg.norm(actions, axis=1)
    return (
        1.0
        - position_error
        - 0.15 * speed
        - 0.1 * yaw_error
        - 1.5 * clearance_penalty
        - 0.02 * control
    ).astype(np.float32)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
