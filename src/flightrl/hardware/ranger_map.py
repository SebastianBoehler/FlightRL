from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Iterable, Mapping

import numpy as np


RANGER_KEYS = ("range.front", "range.back", "range.left", "range.right", "range.up", "range.zrange")
HORIZONTAL_RANGER_KEYS = ("range.front", "range.back", "range.left", "range.right")
BODY_DIRECTIONS = {
    "range.front": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    "range.back": np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
    "range.left": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    "range.right": np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
    "range.up": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    "range.zrange": np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
}


@dataclass(frozen=True, slots=True)
class RangerPoint:
    time_s: float
    sensor: str
    drone_x_m: float
    drone_y_m: float
    drone_z_m: float
    x_m: float
    y_m: float
    z_m: float
    distance_m: float


@dataclass(frozen=True, slots=True)
class DronePose:
    time_s: float
    x_m: float
    y_m: float
    z_m: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


def points_from_rows(
    rows: Iterable[Mapping[str, str | float]],
    *,
    max_range_m: float = 4.0,
    min_range_m: float = 0.03,
) -> list[RangerPoint]:
    points: list[RangerPoint] = []
    for row in rows:
        position = np.asarray(
            [_float(row, "stateEstimate.x"), _float(row, "stateEstimate.y"), _float(row, "stateEstimate.z")],
            dtype=np.float32,
        )
        rotation = euler_matrix(
            _deg_to_rad(_float(row, "stabilizer.roll")),
            _deg_to_rad(_float(row, "stabilizer.pitch")),
            _deg_to_rad(_float(row, "stabilizer.yaw")),
        )
        time_s = _float(row, "host_time_s")
        for key in RANGER_KEYS:
            distance_m = _float(row, key) / 1000.0
            if not min_range_m <= distance_m <= max_range_m:
                continue
            point = position + rotation @ BODY_DIRECTIONS[key] * distance_m
            points.append(
                RangerPoint(
                    time_s=time_s,
                    sensor=key,
                    drone_x_m=float(position[0]),
                    drone_y_m=float(position[1]),
                    drone_z_m=float(position[2]),
                    x_m=float(point[0]),
                    y_m=float(point[1]),
                    z_m=float(point[2]),
                    distance_m=float(distance_m),
                )
            )
    return points


def trajectory_from_rows(rows: Iterable[Mapping[str, str | float]]) -> list[DronePose]:
    poses: list[DronePose] = []
    for row in rows:
        poses.append(
            DronePose(
                time_s=_float(row, "host_time_s"),
                x_m=_float(row, "stateEstimate.x"),
                y_m=_float(row, "stateEstimate.y"),
                z_m=_float(row, "stateEstimate.z"),
                roll_deg=_float(row, "stabilizer.roll"),
                pitch_deg=_float(row, "stabilizer.pitch"),
                yaw_deg=_float(row, "stabilizer.yaw"),
            )
        )
    return poses


def prepare_rows(
    rows: Iterable[Mapping[str, str | float]],
    *,
    min_drone_z_m: float = 0.0,
    normalize_xy: bool = True,
) -> list[dict[str, str | float]]:
    filtered = [dict(row) for row in rows if _float(row, "stateEstimate.z") >= min_drone_z_m]
    if not filtered:
        return []
    t0 = _float(filtered[0], "host_time_s")
    x0 = _float(filtered[0], "stateEstimate.x") if normalize_xy else 0.0
    y0 = _float(filtered[0], "stateEstimate.y") if normalize_xy else 0.0
    for row in filtered:
        row["host_time_s"] = str(_float(row, "host_time_s") - t0)
        row["stateEstimate.x"] = str(_float(row, "stateEstimate.x") - x0)
        row["stateEstimate.y"] = str(_float(row, "stateEstimate.y") - y0)
    return filtered


def summarize_map(
    points: list[RangerPoint],
    trajectory: list[DronePose],
    *,
    min_points: int = 100,
    min_duration_s: float = 10.0,
    min_horizontal_sensors: int = 3,
    min_trajectory_xy_span_m: float = 0.25,
) -> dict:
    sensor_counts = {key: sum(1 for point in points if point.sensor == key) for key in RANGER_KEYS}
    active_horizontal = [key for key in HORIZONTAL_RANGER_KEYS if sensor_counts[key] > 0]
    point_xyz = np.asarray([[point.x_m, point.y_m, point.z_m] for point in points], dtype=np.float32)
    pose_xyz = np.asarray([[pose.x_m, pose.y_m, pose.z_m] for pose in trajectory], dtype=np.float32)
    duration = trajectory[-1].time_s - trajectory[0].time_s if len(trajectory) >= 2 else 0.0
    summary = {
        "point_count": len(points),
        "pose_count": len(trajectory),
        "duration_s": float(max(duration, 0.0)),
        "points_per_second": float(len(points) / max(duration, 1e-6)),
        "sensor_counts": sensor_counts,
        "active_horizontal_sensors": active_horizontal,
        "point_cloud": bounds_summary(point_xyz),
        "trajectory": bounds_summary(pose_xyz),
        "trajectory_path_length_m": path_length(pose_xyz),
    }
    failures = []
    if summary["point_count"] < min_points:
        failures.append("points")
    if summary["duration_s"] < min_duration_s:
        failures.append("duration")
    if len(active_horizontal) < min_horizontal_sensors:
        failures.append("horizontal_sensor_coverage")
    if summary["trajectory"]["xy_span_m"] < min_trajectory_xy_span_m:
        failures.append("trajectory_xy_span")
    summary["mapping_ready"] = not failures
    summary["failures"] = failures
    return summary


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


def euler_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def _float(row: Mapping[str, str | float], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _deg_to_rad(value: float) -> float:
    return value * pi / 180.0
