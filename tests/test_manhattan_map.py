from __future__ import annotations

import numpy as np

from flightrl.hardware.manhattan_map import fit_manhattan_box, rotate_xy, snap_points_to_box


def test_fit_manhattan_box_recovers_rotated_rectangle() -> None:
    rng = np.random.default_rng(7)
    local = rectangle_wall_points(rng)
    angle = 0.33
    world_xy = rotate_xy(local[:, :2], angle)
    points = np.column_stack([world_xy, local[:, 2]]).astype(np.float32)

    fit = fit_manhattan_box(points, angle_samples=360, quantile=0.01, max_wall_residual_m=0.12)

    assert abs(fit.angle_rad - angle) < 0.03
    assert abs((fit.x_max - fit.x_min) - 4.0) < 0.20
    assert abs((fit.y_max - fit.y_min) - 2.5) < 0.20
    assert fit.wall_fraction > 0.95


def test_snap_points_to_box_drops_large_residual_outliers() -> None:
    rng = np.random.default_rng(11)
    points = rectangle_wall_points(rng)
    outlier = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    fit = fit_manhattan_box(np.concatenate([points, outlier]), quantile=0.01, max_wall_residual_m=0.12)

    snapped = snap_points_to_box(np.concatenate([points, outlier]), fit, max_wall_residual_m=0.12)

    assert len(snapped) == len(points)


def rectangle_wall_points(rng: np.random.Generator) -> np.ndarray:
    z = rng.uniform(0.1, 2.2, 320)
    x_edge = rng.choice([-2.0, 2.0], 160) + rng.normal(0.0, 0.02, 160)
    y_edge = rng.choice([-1.25, 1.25], 160) + rng.normal(0.0, 0.02, 160)
    x_points = np.column_stack([x_edge, rng.uniform(-1.25, 1.25, 160), z[:160]])
    y_points = np.column_stack([rng.uniform(-2.0, 2.0, 160), y_edge, z[160:]])
    return np.concatenate([x_points, y_points]).astype(np.float32)
