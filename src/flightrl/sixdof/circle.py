from __future__ import annotations

import numpy as np


def circle_tangent_yaw_from_arrays(
    position: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    radial = position[:, :2] - target_position[:, :2]
    radius = np.maximum(np.linalg.norm(radial[:, :2], axis=1, keepdims=True), 0.2)
    tangent = np.concatenate([-radial[:, 1:2], radial[:, 0:1]], axis=1) / radius
    return np.arctan2(tangent[:, 1], tangent[:, 0]).astype(np.float32)


def circle_orbit_error_from_arrays(
    position: np.ndarray,
    target_position: np.ndarray,
    target_radius_m: float = 0.75,
) -> np.ndarray:
    radial = position - target_position
    radius_error = np.linalg.norm(radial[:, :2], axis=1) - target_radius_m
    z_error = position[:, 2] - target_position[:, 2]
    return np.sqrt(radius_error * radius_error + z_error * z_error).astype(np.float32)
