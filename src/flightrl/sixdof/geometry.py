from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SENSOR_NAMES = ("front", "back", "left", "right", "up", "down")
BODY_RAYS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class BoxRoom:
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0
    z_min: float = 0.0
    z_max: float = 2.5
    max_range_m: float = 4.0

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        return ((self.x_min, self.x_max), (self.y_min, self.y_max), (self.z_min, self.z_max))

    def contains(self, positions: np.ndarray, margin: float = 0.03) -> np.ndarray:
        x_ok = (positions[:, 0] >= self.x_min + margin) & (positions[:, 0] <= self.x_max - margin)
        y_ok = (positions[:, 1] >= self.y_min + margin) & (positions[:, 1] <= self.y_max - margin)
        z_ok = (positions[:, 2] >= self.z_min + margin) & (positions[:, 2] <= self.z_max - margin)
        return x_ok & y_ok & z_ok

    def raycast(self, positions: np.ndarray, directions: np.ndarray) -> np.ndarray:
        distances = np.full(positions.shape[0], self.max_range_m, dtype=np.float32)
        eps = 1e-6
        for axis, (low, high) in enumerate(self.bounds):
            for plane in (low, high):
                denom = directions[:, axis]
                active = np.abs(denom) > eps
                t = np.full(positions.shape[0], np.inf, dtype=np.float32)
                t[active] = (plane - positions[active, axis]) / denom[active]
                hit = t > eps
                for other_axis, (other_low, other_high) in enumerate(self.bounds):
                    if other_axis == axis:
                        continue
                    coord = np.full(positions.shape[0], np.inf, dtype=np.float32)
                    coord[active] = positions[active, other_axis] + t[active] * directions[active, other_axis]
                    hit &= (coord >= other_low - eps) & (coord <= other_high + eps)
                distances = np.where(hit & (t < distances), t, distances)
        return np.clip(distances, 0.0, self.max_range_m)


def body_rays_world(quaternions: np.ndarray) -> np.ndarray:
    rotation = quat_to_matrix(quaternions)
    return np.einsum("nij,kj->nki", rotation, BODY_RAYS, optimize=True)


def quat_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    q = normalize_quat(quaternions)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrix = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    matrix[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[:, 0, 1] = 2.0 * (x * y - z * w)
    matrix[:, 0, 2] = 2.0 * (x * z + y * w)
    matrix[:, 1, 0] = 2.0 * (x * y + z * w)
    matrix[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[:, 1, 2] = 2.0 * (y * z - x * w)
    matrix[:, 2, 0] = 2.0 * (x * z - y * w)
    matrix[:, 2, 1] = 2.0 * (y * z + x * w)
    matrix[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def normalize_quat(quaternions: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions / np.maximum(norm, 1e-8)
