from __future__ import annotations

import numpy as np


CONTROLLERS = ("policy", "teacher_residual")


def validate_controller(controller: str) -> str:
    if controller not in CONTROLLERS:
        raise ValueError(f"unknown 6-DoF controller {controller!r}")
    return controller


def executed_action_for_controller(
    controller: str,
    actor_action: np.ndarray,
    teacher_action: np.ndarray,
    residual_scale: float,
) -> np.ndarray:
    validate_controller(controller)
    if controller == "teacher_residual":
        return np.clip(teacher_action + residual_scale * actor_action, -1.0, 1.0).astype(np.float32)
    return actor_action.astype(np.float32)


def imitation_target_for_controller(controller: str, teacher_action: np.ndarray) -> np.ndarray:
    validate_controller(controller)
    if controller == "teacher_residual":
        return np.zeros_like(teacher_action, dtype=np.float32)
    return teacher_action.astype(np.float32)
