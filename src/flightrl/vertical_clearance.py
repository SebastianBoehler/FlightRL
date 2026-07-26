from __future__ import annotations

from typing import Protocol

import numpy as np


LIVE_VERTICAL_TOP_CLEARANCE_M = 0.45
LIVE_VERTICAL_BOTTOM_CLEARANCE_M = 0.45
LIVE_VERTICAL_FLOOR_GUARD_M = 0.30


class VerticalRangeLike(Protocol):
    up_m: float
    zrange_m: float


def vertical_velocity_from_clearance(
    reading: VerticalRangeLike,
    *,
    top_clearance_m: float = LIVE_VERTICAL_TOP_CLEARANCE_M,
    bottom_clearance_m: float = LIVE_VERTICAL_BOTTOM_CLEARANCE_M,
    hard_clearance_m: float = 0.10,
    max_vertical_speed_m_s: float = 0.18,
    bottom_floor_guard_m: float = LIVE_VERTICAL_FLOOR_GUARD_M,
) -> float:
    push = vertical_clearance_push(
        reading.up_m,
        reading.zrange_m,
        top_clearance_m=top_clearance_m,
        bottom_clearance_m=bottom_clearance_m,
        hard_clearance_m=hard_clearance_m,
        bottom_floor_guard_m=bottom_floor_guard_m,
    )
    return float(np.clip(max_vertical_speed_m_s * push, -max_vertical_speed_m_s, max_vertical_speed_m_s))


def vertical_clearance_push(
    up_m: float,
    zrange_m: float,
    *,
    top_clearance_m: float = LIVE_VERTICAL_TOP_CLEARANCE_M,
    bottom_clearance_m: float = LIVE_VERTICAL_BOTTOM_CLEARANCE_M,
    hard_clearance_m: float = 0.10,
    bottom_floor_guard_m: float = LIVE_VERTICAL_FLOOR_GUARD_M,
) -> float:
    push = _pressure(zrange_m, bottom_clearance_m, hard_clearance_m) - _pressure(up_m, top_clearance_m, hard_clearance_m)
    if zrange_m <= bottom_floor_guard_m:
        push = max(push, _pressure(zrange_m, bottom_clearance_m, hard_clearance_m))
    return float(np.clip(push, -1.0, 1.0))


def vertical_clearance_push_np(
    up_m: np.ndarray,
    zrange_m: np.ndarray,
    *,
    top_clearance_m: float = LIVE_VERTICAL_TOP_CLEARANCE_M,
    bottom_clearance_m: float = LIVE_VERTICAL_BOTTOM_CLEARANCE_M,
    hard_clearance_m: float = 0.10,
    bottom_floor_guard_m: float = LIVE_VERTICAL_FLOOR_GUARD_M,
) -> np.ndarray:
    push = _pressure_np(zrange_m, bottom_clearance_m, hard_clearance_m) - _pressure_np(up_m, top_clearance_m, hard_clearance_m)
    floor_push = _pressure_np(zrange_m, bottom_clearance_m, hard_clearance_m)
    return np.clip(np.where(zrange_m <= bottom_floor_guard_m, np.maximum(push, floor_push), push), -1.0, 1.0)


def _pressure(distance_m: float, clearance_m: float, hard_clearance_m: float) -> float:
    if clearance_m <= hard_clearance_m:
        raise ValueError("clearance_m must be greater than hard_clearance_m")
    scaled = np.clip((clearance_m - distance_m) / (clearance_m - hard_clearance_m), 0.0, 1.0)
    return float(np.sqrt(scaled))


def _pressure_np(distance_m: np.ndarray, clearance_m: float, hard_clearance_m: float) -> np.ndarray:
    if clearance_m <= hard_clearance_m:
        raise ValueError("clearance_m must be greater than hard_clearance_m")
    scaled = np.clip((clearance_m - distance_m) / (clearance_m - hard_clearance_m), 0.0, 1.0)
    return np.sqrt(scaled)
