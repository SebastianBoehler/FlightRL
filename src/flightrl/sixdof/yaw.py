from __future__ import annotations

import numpy as np

from .circle import circle_tangent_yaw_from_arrays
from .env import SixDofCrazyflieEnv, quat_to_yaw, wrap_angle


def yaw_error_for_task(env: SixDofCrazyflieEnv, task: str) -> np.ndarray:
    return np.abs(wrap_angle(target_yaw_for_task(env, task) - quat_to_yaw(env.quaternion))).astype(np.float32)


def yaw_error_for_task_indices(env: SixDofCrazyflieEnv, tasks: tuple[str, ...], task_indices: np.ndarray) -> np.ndarray:
    errors = np.zeros(env.num_envs, dtype=np.float32)
    for index, task in enumerate(tasks):
        mask = task_indices == index
        if np.any(mask):
            errors[mask] = yaw_error_for_task(env, task)[mask]
    return errors


def target_yaw_for_task(env: SixDofCrazyflieEnv, task: str) -> np.ndarray:
    if task == "circle":
        return circle_tangent_yaw(env)
    return env.target_yaw


def circle_tangent_yaw(env: SixDofCrazyflieEnv) -> np.ndarray:
    return circle_tangent_yaw_from_arrays(env.position, env.target_position)
