from __future__ import annotations

from typing import Mapping

import numpy as np


REQUIRED_COMMAND_MODES = ("line_x_pos", "line_x_neg", "line_y_pos", "line_y_neg", "yaw_pos", "yaw_neg")
REQUIRED_COLUMNS = (
    "host_time_s",
    "mode",
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
    "yawrate_deg_s",
    "range.zrange",
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "stabilizer.yaw",
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
)


def summarize_calibration_log(
    rows: list[Mapping[str, str]],
    *,
    min_duration_s: float = 8.0,
    min_rows: int = 100,
    min_floor_valid_ratio: float = 0.5,
    min_yaw_span_deg: float = 45.0,
) -> dict:
    columns = set(rows[0]) if rows else set()
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    modes = sorted({str(row.get("mode", "")) for row in rows if row.get("mode")})
    duration = duration_s(rows)
    monotonic = time_monotonic(rows)
    floor_ratio = valid_range_ratio(rows, "range.zrange")
    yaw_span = angular_span_deg([_float(row, "stabilizer.yaw") for row in rows])
    command_axes = commanded_axes(rows)
    failures = []
    if len(rows) < min_rows:
        failures.append("rows")
    if duration < min_duration_s:
        failures.append("duration")
    if missing_columns:
        failures.append("missing_columns")
    if not monotonic:
        failures.append("time_monotonic")
    if floor_ratio < min_floor_valid_ratio:
        failures.append("floor_range")
    if yaw_span < min_yaw_span_deg:
        failures.append("yaw_span")
    missing_modes = [mode for mode in REQUIRED_COMMAND_MODES if mode not in modes]
    if missing_modes:
        failures.append("command_modes")
    return {
        "rows": len(rows),
        "duration_s": duration,
        "sample_rate_hz": sample_rate_hz(rows),
        "time_monotonic": monotonic,
        "missing_columns": missing_columns,
        "modes": modes,
        "missing_modes": missing_modes,
        "command_axes": command_axes,
        "floor_valid_ratio": floor_ratio,
        "front_valid_ratio": valid_range_ratio(rows, "range.front"),
        "left_valid_ratio": valid_range_ratio(rows, "range.left"),
        "yaw_span_deg": yaw_span,
        "z_span_m": span(rows, "stateEstimate.z"),
        "xy_span_m": xy_span(rows),
        "replay_calibration_ready": not failures,
        "failures": failures,
    }


def commanded_axes(rows: list[Mapping[str, str]]) -> dict[str, bool]:
    return {
        "vx_pos": any(_float(row, "vx_m_s") > 0.0 for row in rows),
        "vx_neg": any(_float(row, "vx_m_s") < 0.0 for row in rows),
        "vy_pos": any(_float(row, "vy_m_s") > 0.0 for row in rows),
        "vy_neg": any(_float(row, "vy_m_s") < 0.0 for row in rows),
        "yaw_pos": any(_float(row, "yawrate_deg_s") > 0.0 for row in rows),
        "yaw_neg": any(_float(row, "yawrate_deg_s") < 0.0 for row in rows),
    }


def duration_s(rows: list[Mapping[str, str]]) -> float:
    if len(rows) < 2:
        return 0.0
    return max(_float(rows[-1], "host_time_s") - _float(rows[0], "host_time_s"), 0.0)


def sample_rate_hz(rows: list[Mapping[str, str]]) -> float:
    duration = duration_s(rows)
    return float((len(rows) - 1) / duration) if len(rows) > 1 and duration > 0.0 else 0.0


def time_monotonic(rows: list[Mapping[str, str]]) -> bool:
    if len(rows) < 2:
        return bool(rows)
    times = [_float(row, "host_time_s") for row in rows]
    return all(curr > prev for prev, curr in zip(times, times[1:]))


def valid_range_ratio(rows: list[Mapping[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    values = np.asarray([_float(row, key) for row in rows], dtype=np.float32)
    return float(np.mean((values > 20.0) & (values < 4000.0)))


def angular_span_deg(values: list[float]) -> float:
    if not values:
        return 0.0
    radians = np.unwrap(np.deg2rad(values))
    return float(np.rad2deg(np.ptp(radians)))


def span(rows: list[Mapping[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    values = np.asarray([_float(row, key) for row in rows], dtype=np.float32)
    return float(np.ptp(values))


def xy_span(rows: list[Mapping[str, str]]) -> float:
    if not rows:
        return 0.0
    xy = np.asarray([[_float(row, "stateEstimate.x"), _float(row, "stateEstimate.y")] for row in rows], dtype=np.float32)
    return float(np.linalg.norm(np.ptp(xy, axis=0)))


def _float(row: Mapping[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
