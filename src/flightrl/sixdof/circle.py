from __future__ import annotations

import numpy as np


def circle_tangent_yaw_from_arrays(position: np.ndarray, target_position: np.ndarray, center_z: float = 0.65) -> np.ndarray:
    center = target_position.copy()
    center[:, 2] = center_z
    radial = position - center
    radial[:, 2] = 0.0
    radius = np.maximum(np.linalg.norm(radial[:, :2], axis=1, keepdims=True), 0.2)
    tangent = np.concatenate([-radial[:, 1:2], radial[:, 0:1]], axis=1) / radius
    return np.arctan2(tangent[:, 1], tangent[:, 0]).astype(np.float32)


def circle_orbit_error_from_arrays(
    position: np.ndarray,
    target_position: np.ndarray,
    target_radius_m: float = 0.75,
    target_z_m: float = 0.65,
) -> np.ndarray:
    radial = position - target_position
    radius_error = np.linalg.norm(radial[:, :2], axis=1) - target_radius_m
    z_error = position[:, 2] - target_z_m
    return np.sqrt(radius_error * radius_error + z_error * z_error).astype(np.float32)
