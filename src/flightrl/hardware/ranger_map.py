from __future__ import annotations

from math import isfinite
from typing import Mapping

import numpy as np

from .map_quality import trajectory_quality
from .ranger_projection import (
    DronePose,
    RangerPoint,
    drone_pose_is_finite,
    ranger_point_is_finite,
)
from .ranger_schema import HORIZONTAL_RANGER_KEYS, RANGER_KEYS


def summarize_map(
    points: list[RangerPoint],
    trajectory: list[DronePose],
    *,
    min_points: int = 100,
    min_duration_s: float = 10.0,
    min_horizontal_sensors: int = 3,
    min_trajectory_xy_span_m: float = 0.25,
    min_yaw_span_deg: float = 0.0,
    max_step_speed_m_s: float = 0.0,
    source_integrity: Mapping[str, object] | None = None,
) -> dict:
    validate_map_thresholds(
        min_points=min_points,
        min_duration_s=min_duration_s,
        min_horizontal_sensors=min_horizontal_sensors,
        min_trajectory_xy_span_m=min_trajectory_xy_span_m,
        min_yaw_span_deg=min_yaw_span_deg,
        max_step_speed_m_s=max_step_speed_m_s,
    )
    invalid_points = sum(not ranger_point_is_finite(point) for point in points)
    invalid_poses = sum(not drone_pose_is_finite(pose) for pose in trajectory)
    points = [point for point in points if ranger_point_is_finite(point)]
    trajectory = [pose for pose in trajectory if drone_pose_is_finite(pose)]
    sensor_counts = {key: sum(1 for point in points if point.sensor == key) for key in RANGER_KEYS}
    active_horizontal = [key for key in HORIZONTAL_RANGER_KEYS if sensor_counts[key] > 0]
    point_xyz = np.asarray([[point.x_m, point.y_m, point.z_m] for point in points], dtype=np.float32)
    pose_xyz = np.asarray([[pose.x_m, pose.y_m, pose.z_m] for pose in trajectory], dtype=np.float32)
    duration = trajectory[-1].time_s - trajectory[0].time_s if len(trajectory) >= 2 else 0.0
    trajectory_path_length = path_length(pose_xyz)
    summary = {
        "point_count": len(points),
        "pose_count": len(trajectory),
        "invalid_point_count": invalid_points,
        "invalid_pose_count": invalid_poses,
        "duration_s": float(max(duration, 0.0)),
        "points_per_second": float(len(points) / max(duration, 1e-6)),
        "sensor_counts": sensor_counts,
        "active_horizontal_sensors": active_horizontal,
        "point_cloud": bounds_summary(point_xyz),
        "trajectory": bounds_summary(pose_xyz),
        "trajectory_path_length_m": trajectory_path_length,
        "trajectory_quality": trajectory_quality(trajectory, pose_xyz, duration, trajectory_path_length, max_step_speed_m_s),
        "source_integrity": dict(source_integrity) if source_integrity is not None else {"present": False},
    }
    summary["point_density_per_path_m"] = float(len(points) / max(summary["trajectory_path_length_m"], 1e-6))
    failures = []
    if invalid_points:
        failures.append("nonfinite_points")
    if invalid_poses:
        failures.append("nonfinite_trajectory")
    if source_integrity is not None and source_integrity.get("valid") is not True:
        failures.extend(str(value) for value in source_integrity.get("failures", ()))
    if summary["point_count"] < min_points:
        failures.append("points")
    if summary["duration_s"] < min_duration_s:
        failures.append("duration")
    if len(active_horizontal) < min_horizontal_sensors:
        failures.append("horizontal_sensor_coverage")
    if summary["trajectory"]["xy_span_m"] < min_trajectory_xy_span_m:
        failures.append("trajectory_xy_span")
    if summary["trajectory_quality"]["yaw_span_deg"] < min_yaw_span_deg:
        failures.append("yaw_span")
    if max_step_speed_m_s > 0.0 and summary["trajectory_quality"]["speed_glitch_count"] > 0:
        failures.append("speed_glitch")
    if not all(
        current.time_s > previous.time_s
        for previous, current in zip(trajectory, trajectory[1:])
    ):
        failures.append("trajectory_time_monotonic")
    summary["mapping_ready"] = not failures
    summary["failures"] = failures
    return summary


def estimate_room_bounds(
    points: list[RangerPoint],
    trajectory: list[DronePose] | None = None,
    *,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
    padding_m: float = 0.05,
    floor_m: float = 0.0,
    max_range_m: float = 4.0,
) -> dict:
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    numeric = (padding_m, floor_m, max_range_m)
    if any(not isfinite(float(value)) for value in numeric) or padding_m < 0.0 or max_range_m <= 0.0:
        raise ValueError("room-bound parameters must be finite and physically valid")
    if any(not ranger_point_is_finite(point) for point in points):
        raise ValueError("room bounds require finite ranger points")
    if trajectory and any(not drone_pose_is_finite(pose) for pose in trajectory):
        raise ValueError("room bounds require finite trajectory poses")
    horizontal = [point for point in points if point.sensor in HORIZONTAL_RANGER_KEYS]
    source = horizontal if len(horizontal) >= 4 else points
    warnings = []
    if len(horizontal) < 4:
        warnings.append("weak_horizontal_coverage")
    x_min, x_max = quantile_bounds([point.x_m for point in source], lower_quantile, upper_quantile, padding_m)
    y_min, y_max = quantile_bounds([point.y_m for point in source], lower_quantile, upper_quantile, padding_m)

    down_points = [point.z_m for point in points if point.sensor == "range.zrange"]
    up_points = [point.z_m for point in points if point.sensor == "range.up"]
    z_min = max(quantile_bounds(down_points, lower_quantile, upper_quantile, padding_m)[0], floor_m) if down_points else floor_m
    if not down_points:
        warnings.append("floor_from_default")
    z_source = up_points or [point.z_m for point in points]
    if trajectory:
        z_source = [*z_source, *[pose.z_m for pose in trajectory]]
    z_max = quantile_bounds(z_source, lower_quantile, upper_quantile, padding_m)[1]
    if not up_points:
        warnings.append("ceiling_from_non_up_points")

    bounds = enforce_min_span(
        {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "z_min": z_min, "z_max": z_max},
        min_span_m=0.1,
    )
    if bounds["z_min"] < floor_m:
        bounds["z_max"] += floor_m - bounds["z_min"]
        bounds["z_min"] = floor_m
    return {
        **bounds,
        "width_m": bounds["x_max"] - bounds["x_min"],
        "depth_m": bounds["y_max"] - bounds["y_min"],
        "height_m": bounds["z_max"] - bounds["z_min"],
        "max_range_m": max_range_m,
        "point_count": len(points),
        "horizontal_point_count": len(horizontal),
        "up_point_count": len(up_points),
        "down_point_count": len(down_points),
        "method": "axis_aligned_quantile_box",
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "padding_m": padding_m,
        "warnings": warnings,
    }


def quantile_bounds(values: list[float], lower_quantile: float, upper_quantile: float, padding_m: float) -> tuple[float, float]:
    if not values:
        return -padding_m, padding_m
    if any(not isfinite(value) for value in values):
        raise ValueError("quantile inputs must be finite")
    array = np.asarray(values, dtype=np.float32)
    return float(np.quantile(array, lower_quantile) - padding_m), float(np.quantile(array, upper_quantile) + padding_m)


def enforce_min_span(bounds: dict[str, float], *, min_span_m: float) -> dict[str, float]:
    adjusted = dict(bounds)
    for low_key, high_key in (("x_min", "x_max"), ("y_min", "y_max"), ("z_min", "z_max")):
        span = adjusted[high_key] - adjusted[low_key]
        if span >= min_span_m:
            continue
        center = 0.5 * (adjusted[low_key] + adjusted[high_key])
        adjusted[low_key] = center - min_span_m * 0.5
        adjusted[high_key] = center + min_span_m * 0.5
    return adjusted


def bounds_summary(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"x_span_m": 0.0, "y_span_m": 0.0, "z_span_m": 0.0, "xy_span_m": 0.0}
    span = np.ptp(values, axis=0)
    return {
        "x_span_m": float(span[0]),
        "y_span_m": float(span[1]),
        "z_span_m": float(span[2]),
        "xy_span_m": float(np.linalg.norm(span[:2])),
        "z_min_m": float(np.min(values[:, 2])),
        "z_max_m": float(np.max(values[:, 2])),
    }


def path_length(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(values, axis=0), axis=1)))


def validate_map_thresholds(
    *,
    min_points: int,
    min_duration_s: float,
    min_horizontal_sensors: int,
    min_trajectory_xy_span_m: float,
    min_yaw_span_deg: float,
    max_step_speed_m_s: float,
) -> None:
    if type(min_points) is not int or min_points < 0:
        raise ValueError("min_points must be a nonnegative integer")
    if (
        type(min_horizontal_sensors) is not int
        or not 0 <= min_horizontal_sensors <= len(HORIZONTAL_RANGER_KEYS)
    ):
        raise ValueError("min_horizontal_sensors is outside the supported range")
    values = (
        min_duration_s,
        min_trajectory_xy_span_m,
        min_yaw_span_deg,
        max_step_speed_m_s,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or value < 0.0
        for value in values
    ):
        raise ValueError("map thresholds must be finite and nonnegative")
