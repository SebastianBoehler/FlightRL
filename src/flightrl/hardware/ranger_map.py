from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Iterable, Mapping

import numpy as np


RANGER_KEYS = ("range.front", "range.back", "range.left", "range.right", "range.up", "range.zrange")
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
