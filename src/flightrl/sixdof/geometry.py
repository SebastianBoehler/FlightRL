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
class AxisAlignedObstacle:
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0
    z_min: float = 0.0
    z_max: float = 2.5

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        return ((self.x_min, self.x_max), (self.y_min, self.y_max), (self.z_min, self.z_max))

    def contains_points(self, positions: np.ndarray, margin: float = 0.0) -> np.ndarray:
        x_ok = (positions[:, 0] >= self.x_min - margin) & (positions[:, 0] <= self.x_max + margin)
        y_ok = (positions[:, 1] >= self.y_min - margin) & (positions[:, 1] <= self.y_max + margin)
        z_ok = (positions[:, 2] >= self.z_min - margin) & (positions[:, 2] <= self.z_max + margin)
        return x_ok & y_ok & z_ok

    def raycast(self, positions: np.ndarray, directions: np.ndarray, max_range_m: float) -> np.ndarray:
        eps = 1e-6
        lower = np.asarray((self.x_min, self.y_min, self.z_min), dtype=np.float32)
        upper = np.asarray((self.x_max, self.y_max, self.z_max), dtype=np.float32)
        parallel = np.abs(directions) <= eps
        inverse = np.divide(
            1.0,
            directions,
            out=np.zeros_like(directions),
            where=~parallel,
        )
        first = (lower - positions) * inverse
        second = (upper - positions) * inverse
        near = np.minimum(first, second)
        far = np.maximum(first, second)
        inside_parallel = (positions >= lower - eps) & (positions <= upper + eps)
        near = np.where(parallel & inside_parallel, -np.inf, near)
        far = np.where(parallel & inside_parallel, np.inf, far)
        invalid_parallel = np.any(parallel & ~inside_parallel, axis=1)
        enter = np.max(near, axis=1)
        leave = np.min(far, axis=1)
        distance = np.where(enter > eps, enter, leave)
        hit = ~invalid_parallel & (leave >= np.maximum(enter, eps))
        return np.where(hit, np.minimum(distance, max_range_m), max_range_m).astype(
            np.float32
        )


@dataclass(frozen=True, slots=True)
class BoxRoom(AxisAlignedObstacle):
    max_range_m: float = 4.0
    obstacles: tuple[AxisAlignedObstacle, ...] = ()

    def contains(self, positions: np.ndarray, margin: float = 0.03) -> np.ndarray:
        inside_room = (
            (positions[:, 0] >= self.x_min + margin)
            & (positions[:, 0] <= self.x_max - margin)
            & (positions[:, 1] >= self.y_min + margin)
            & (positions[:, 1] <= self.y_max - margin)
            & (positions[:, 2] >= self.z_min + margin)
            & (positions[:, 2] <= self.z_max - margin)
        )
        for obstacle in self.obstacles:
            inside_room &= ~obstacle.contains_points(positions, margin=margin)
        return inside_room

    def raycast(self, positions: np.ndarray, directions: np.ndarray) -> np.ndarray:
        distances = AxisAlignedObstacle.raycast(self, positions, directions, self.max_range_m)
        for obstacle in self.obstacles:
            distances = np.minimum(distances, obstacle.raycast(positions, directions, self.max_range_m))
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
