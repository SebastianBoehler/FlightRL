from __future__ import annotations

import numpy as np

from .geometry import quat_to_matrix


TARGET_SHAPINGS = ("none", "precontact_drift_brake")


def shape_action_targets(env, targets: np.ndarray, mode: str, strength: float = 1.0) -> np.ndarray:
    if mode == "none":
        return np.asarray(targets, dtype=np.float32).copy()
    if mode != "precontact_drift_brake":
        raise ValueError(f"unknown target shaping {mode!r}; expected one of {', '.join(TARGET_SHAPINGS)}")
    shaped = np.asarray(targets, dtype=np.float32).copy()
    min_clearance = np.min(env.ranges_m[:, :4], axis=1)
    open_weight = np.clip((min_clearance - 0.50) / 0.25, 0.0, 1.0)
    body_velocity = body_frame_velocity(env.quaternion, env.velocity)
    body_horizontal = body_velocity[:, :2]
    body_speed = np.linalg.norm(body_horizontal, axis=1)
    speed_weight = np.clip((body_speed - 0.25) / 0.75, 0.0, 1.0)
    weight = (np.clip(strength, 0.0, 1.0) * open_weight * speed_weight).astype(np.float32)
    brake_direction = -body_horizontal / np.maximum(body_speed[:, None], 0.05)
    brake_magnitude = np.clip(0.38 * body_speed, 0.0, 0.70)
    brake_control = brake_direction * brake_magnitude[:, None]
    brake = shaped.copy()
    brake[:, 2] = brake_control[:, 0]
    brake[:, 1] = -brake_control[:, 1]
    shaped[:, 1:3] = (1.0 - weight[:, None]) * shaped[:, 1:3] + weight[:, None] * brake[:, 1:3]
    shaped[:, 1:3] = np.clip(shaped[:, 1:3], -0.70, 0.70)
    return shaped.astype(np.float32)


def body_frame_velocity(quaternions: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    return np.einsum("nij,ni->nj", quat_to_matrix(quaternions), velocity, optimize=True).astype(np.float32)
