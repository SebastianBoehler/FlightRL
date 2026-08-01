from __future__ import annotations

import numpy as np


TASKS = ("position_yaw", "obstacle_avoidance", "circle")
MULTITASK = "multitask"


def parse_task_spec(value: str) -> tuple[str, ...]:
    if value == MULTITASK:
        return TASKS
    tasks = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = [task for task in tasks if task not in TASKS]
    if invalid:
        raise ValueError(f"unknown 6-DoF task(s): {', '.join(invalid)}")
    if not tasks:
        raise ValueError("at least one 6-DoF task is required")
    return tasks


def task_observation_dim(tasks: tuple[str, ...]) -> int:
    return len(tasks) if len(tasks) > 1 else 0


def task_indices_for_name(task: str, tasks: tuple[str, ...], count: int) -> np.ndarray:
    if task not in tasks:
        raise ValueError(f"task {task!r} is not in checkpoint tasks {tasks}")
    return np.full(count, tasks.index(task), dtype=np.int64)


def append_task_encoding(observations: np.ndarray, task_indices: np.ndarray, task_count: int) -> np.ndarray:
    if task_count <= 1:
        return observations
    one_hot = np.zeros((observations.shape[0], task_count), dtype=np.float32)
    one_hot[np.arange(observations.shape[0]), task_indices] = 1.0
    return np.concatenate([observations, one_hot], axis=1).astype(np.float32)


def select_task_actions(action_by_task: dict[str, np.ndarray], task_indices: np.ndarray, tasks: tuple[str, ...]) -> np.ndarray:
    first = next(iter(action_by_task.values()))
    actions = np.zeros_like(first)
    for index, task in enumerate(tasks):
        mask = task_indices == index
        if np.any(mask):
            actions[mask] = action_by_task[task][mask]
    return actions
