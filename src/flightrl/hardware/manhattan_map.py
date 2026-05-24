from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HORIZONTAL_SENSORS = {"range.front", "range.back", "range.left", "range.right"}


@dataclass(frozen=True, slots=True)
class ManhattanFit:
    angle_rad: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    median_wall_residual_m: float
    wall_fraction: float
    point_count: int


def fit_manhattan_box(points: np.ndarray, *, angle_samples: int = 181, quantile: float = 0.03, max_wall_residual_m: float = 0.35) -> ManhattanFit:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(points) < 8:
        raise ValueError("at least 8 points are required for Manhattan fitting")
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must be between 0 and 0.5")
    if angle_samples < 4:
        raise ValueError("angle_samples must be >= 4")

    xy = points[:, :2].astype(np.float64)
    best_score: tuple[float, float] | None = None
    best_fit: ManhattanFit | None = None
    for angle in np.linspace(0.0, np.pi / 2.0, angle_samples, endpoint=False):
        rotated = rotate_xy(xy, -angle)
        x_min, x_max = quantile_pair(rotated[:, 0], quantile)
        y_min, y_max = quantile_pair(rotated[:, 1], quantile)
        residual = wall_residual(rotated, x_min, x_max, y_min, y_max)
        median = float(np.median(residual))
        wall_fraction = float(np.mean(residual <= max_wall_residual_m))
        score = (median, -wall_fraction)
        if best_score is None or score < best_score:
            best_score = score
            z_min, z_max = quantile_pair(points[:, 2], quantile)
            best_fit = ManhattanFit(
                angle_rad=float(angle),
                x_min=float(x_min),
                x_max=float(x_max),
                y_min=float(y_min),
                y_max=float(y_max),
                z_min=float(z_min),
                z_max=float(z_max),
                median_wall_residual_m=median,
                wall_fraction=wall_fraction,
                point_count=int(len(points)),
            )
    assert best_fit is not None
    return best_fit


def snap_points_to_box(points: np.ndarray, fit: ManhattanFit, *, max_wall_residual_m: float = 0.35) -> np.ndarray:
    rotated = rotate_xy(points[:, :2], -fit.angle_rad)
    bounds = np.asarray([fit.x_min, fit.x_max, fit.y_min, fit.y_max], dtype=np.float64)
    distances = np.column_stack(
        [
            np.abs(rotated[:, 0] - fit.x_min),
            np.abs(rotated[:, 0] - fit.x_max),
            np.abs(rotated[:, 1] - fit.y_min),
            np.abs(rotated[:, 1] - fit.y_max),
        ]
    )
    nearest = np.argmin(distances, axis=1)
    snapped = rotated.copy()
    snapped[nearest == 0, 0] = bounds[0]
    snapped[nearest == 1, 0] = bounds[1]
    snapped[nearest == 2, 1] = bounds[2]
    snapped[nearest == 3, 1] = bounds[3]
    mask = np.min(distances, axis=1) <= max_wall_residual_m
    world = rotate_xy(snapped[mask], fit.angle_rad)
    return np.column_stack([world, points[mask, 2]]).astype(np.float32)


def box_corners_world(fit: ManhattanFit) -> np.ndarray:
    xy = np.asarray(
        [
            [fit.x_min, fit.y_min],
            [fit.x_max, fit.y_min],
            [fit.x_max, fit.y_max],
            [fit.x_min, fit.y_max],
            [fit.x_min, fit.y_min],
        ],
        dtype=np.float64,
    )
    return rotate_xy(xy, fit.angle_rad).astype(np.float32)


def fit_to_dict(fit: ManhattanFit) -> dict[str, float | int]:
    return {
        "method": "manhattan_axis_sweep_quantile_box",
        "angle_rad": fit.angle_rad,
        "angle_deg": float(np.rad2deg(fit.angle_rad)),
        "x_min": fit.x_min,
        "x_max": fit.x_max,
        "y_min": fit.y_min,
        "y_max": fit.y_max,
        "z_min": fit.z_min,
        "z_max": fit.z_max,
        "width_m": fit.x_max - fit.x_min,
        "depth_m": fit.y_max - fit.y_min,
        "height_m": fit.z_max - fit.z_min,
        "median_wall_residual_m": fit.median_wall_residual_m,
        "wall_fraction": fit.wall_fraction,
        "point_count": fit.point_count,
    }


def quantile_pair(values: np.ndarray, quantile: float) -> tuple[float, float]:
    return float(np.quantile(values, quantile)), float(np.quantile(values, 1.0 - quantile))


def wall_residual(rotated_xy: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
    return np.min(
        np.column_stack(
            [
                np.abs(rotated_xy[:, 0] - x_min),
                np.abs(rotated_xy[:, 0] - x_max),
                np.abs(rotated_xy[:, 1] - y_min),
                np.abs(rotated_xy[:, 1] - y_max),
            ]
        ),
        axis=1,
    )


def rotate_xy(xy: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return xy @ rotation.T
