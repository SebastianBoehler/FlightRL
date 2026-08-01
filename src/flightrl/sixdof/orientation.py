from __future__ import annotations

import numpy as np

from .geometry import normalize_quat


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left[:, 0], left[:, 1], left[:, 2], left[:, 3]
    rw, rx, ry, rz = right[:, 0], right[:, 1], right[:, 2], right[:, 3]
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=1,
    ).astype(np.float32)


def euler_to_quat(
    roll: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=1,
    ).astype(np.float32)


def quat_to_yaw(quaternions: np.ndarray) -> np.ndarray:
    q = normalize_quat(quaternions)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    ).astype(np.float32)
