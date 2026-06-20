from __future__ import annotations

from typing import Protocol

import numpy as np


class HorizontalRangeLike(Protocol):
    front_m: float
    back_m: float
    left_m: float
    right_m: float


def min_horizontal_ttc_s(reading: HorizontalRangeLike, range_rate_m_s: HorizontalRangeLike | None) -> float:
    if range_rate_m_s is None:
        return float("inf")
    return min(
        time_to_collision_s(reading.front_m, range_rate_m_s.front_m),
        time_to_collision_s(reading.back_m, range_rate_m_s.back_m),
        time_to_collision_s(reading.left_m, range_rate_m_s.left_m),
        time_to_collision_s(reading.right_m, range_rate_m_s.right_m),
    )


def ttc_escape_pressure(reading: HorizontalRangeLike, rate: HorizontalRangeLike, horizon_s: float, hard_s: float) -> tuple[float, float]:
    front = _ttc_pressure(reading.front_m, rate.front_m, horizon_s, hard_s)
    back = _ttc_pressure(reading.back_m, rate.back_m, horizon_s, hard_s)
    left = _ttc_pressure(reading.left_m, rate.left_m, horizon_s, hard_s)
    right = _ttc_pressure(reading.right_m, rate.right_m, horizon_s, hard_s)
    return back - front, right - left


def _ttc_pressure(distance_m: float, rate_m_s: float, horizon_s: float, hard_s: float) -> float:
    ttc_s = time_to_collision_s(distance_m, rate_m_s)
    if not np.isfinite(ttc_s) or ttc_s >= horizon_s:
        return 0.0
    return float(np.sqrt(np.clip((horizon_s - ttc_s) / (horizon_s - hard_s), 0.0, 1.0)))


def time_to_collision_s(distance_m: float, rate_m_s: float) -> float:
    closing_speed_m_s = max(-rate_m_s, 0.0)
    if closing_speed_m_s <= 1e-6:
        return float("inf")
    return max(distance_m, 0.0) / closing_speed_m_s
