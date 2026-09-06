"""Calibrated pinhole geometry, OpenCV optical coordinates, ray-distance depth."""

import numpy as np


def intrinsics(width=256, height=192):
    f = height / (2 * np.tan(1.099557429 / 2))
    return np.array(
        [[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]], float
    )


def unproject(pixels, distance, k):
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(k).T
    return rays / np.linalg.norm(rays, axis=1)[:, None] * distance[:, None]


def transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def dense_points(depth, k, stride=8):
    y, x = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    pixels = np.c_[x.ravel(), y.ravel()]
    d = depth[y, x].ravel()
    valid = np.isfinite(d) & (d > 0.1) & (d < 7.95)
    return unproject(pixels[valid], d[valid], k), pixels[valid].astype(int)


def axial_depth_points(depth, k, camera_to_world):
    """Backproject camera-Z depth, distinct from the simulator's ray distance."""
    h, w = depth.shape
    y, x = np.mgrid[:h, :w]
    rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.linalg.inv(k).T
    return transform((rays * depth[..., None]).reshape(-1, 3), camera_to_world).reshape(
        h, w, 3
    )
