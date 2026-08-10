from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from typing import Iterable, Mapping

import numpy as np

from .ranger_schema import RANGER_KEYS, RANGER_POSE_KEYS


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
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in (min_range_m, max_range_m)
        )
        or not 0.0 < min_range_m < max_range_m
    ):
        raise ValueError("ranger limits must be finite with 0 < min < max")
    points: list[RangerPoint] = []
    for row in rows:
        pose_values = finite_row_values(row, RANGER_POSE_KEYS)
        if pose_values is None:
            continue
        time_s, x_m, y_m, z_m, roll_deg, pitch_deg, yaw_deg = pose_values
        position = np.asarray([x_m, y_m, z_m], dtype=np.float32)
        rotation = euler_matrix(
            _deg_to_rad(roll_deg),
            _deg_to_rad(pitch_deg),
            _deg_to_rad(yaw_deg),
        )
        for key in RANGER_KEYS:
            distance_mm = finite_row_value(row, key)
            if distance_mm is None:
                continue
            distance_m = distance_mm / 1000.0
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
        values = finite_row_values(row, RANGER_POSE_KEYS)
        if values is not None:
            poses.append(DronePose(*values))
    return poses


def prepare_rows(
    rows: Iterable[Mapping[str, str | float]],
    *,
    min_drone_z_m: float = 0.0,
    normalize_xy: bool = True,
) -> list[dict[str, str | float]]:
    if (
        isinstance(min_drone_z_m, bool)
        or not isinstance(min_drone_z_m, (int, float))
        or not isfinite(float(min_drone_z_m))
        or type(normalize_xy) is not bool
    ):
        raise ValueError("room preprocessing values are invalid")
    filtered = [
        dict(row)
        for row in rows
        if (z_m := finite_row_value(row, "stateEstimate.z")) is not None
        and z_m >= min_drone_z_m
        and finite_row_values(
            row,
            ("host_time_s", "stateEstimate.x", "stateEstimate.y"),
        )
        is not None
    ]
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


def rows_with_mapping_time(
    rows: Iterable[Mapping[str, str | float]],
) -> tuple[list[dict[str, str | float]], str]:
    copied = [dict(row) for row in rows]
    if not copied or not any("crazyflie_time_ms" in row for row in copied):
        return copied, "host_time_s"
    if not all("crazyflie_time_ms" in row for row in copied):
        raise ValueError("mapping device timestamps must be present in every row")
    previous = -1.0
    for row in copied:
        timestamp_ms = finite_row_value(row, "crazyflie_time_ms")
        if (
            timestamp_ms is None
            or not timestamp_ms.is_integer()
            or not 0.0 <= timestamp_ms <= 0xFFFFFFFF
            or timestamp_ms <= previous
        ):
            raise ValueError("mapping device timestamps must be ordered uint32 milliseconds")
        previous = timestamp_ms
        row["host_time_s"] = str(timestamp_ms / 1000.0)
    return copied, "crazyflie_time_ms"


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


def finite_row_value(
    row: Mapping[str, str | float],
    key: str,
) -> float | None:
    try:
        value = float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def finite_row_values(
    row: Mapping[str, str | float],
    keys: tuple[str, ...],
) -> tuple[float, ...] | None:
    values = tuple(finite_row_value(row, key) for key in keys)
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values if value is not None)


def ranger_point_is_finite(point: RangerPoint) -> bool:
    return point.sensor in RANGER_KEYS and point.distance_m > 0.0 and all(
        isfinite(value)
        for value in (
            point.time_s,
            point.drone_x_m,
            point.drone_y_m,
            point.drone_z_m,
            point.x_m,
            point.y_m,
            point.z_m,
            point.distance_m,
        )
    )


def drone_pose_is_finite(pose: DronePose) -> bool:
    return all(
        isfinite(value)
        for value in (
            pose.time_s,
            pose.x_m,
            pose.y_m,
            pose.z_m,
            pose.roll_deg,
            pose.pitch_deg,
            pose.yaw_deg,
        )
    )


def _float(row: Mapping[str, str | float], key: str) -> float:
    value = finite_row_value(row, key)
    return 0.0 if value is None else value


def _deg_to_rad(value: float) -> float:
    return value * pi / 180.0
