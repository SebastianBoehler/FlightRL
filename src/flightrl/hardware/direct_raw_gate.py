from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


ACTION_COLUMNS = ("action_thrust", "action_roll_rate", "action_pitch_rate", "action_yaw_rate")
RATE_COLUMNS = ("roll_rate_deg_s", "pitch_rate_deg_s", "yaw_rate_deg_s")
HORIZONTAL_RANGES = ("range.front", "range.back", "range.left", "range.right")


@dataclass(frozen=True, slots=True)
class DirectRawGateThresholds:
    min_safe_rows: int = 100
    close_range_mm: float = 350.0
    require_source_health: bool = True
    require_commander_pitch_sign: bool = True
    pitch_sign_tolerance_deg_s: float = 1e-4
    precontact_min_horizontal_mm: float = 450.0
    max_precontact_horizontal_speed_m_s: float = 0.45
    max_safe_action_saturation: float = 0.08
    max_close_action_saturation: float = 0.18
    min_thrust_p05_percent: float = 35.0
    max_thrust_p95_percent: float = 68.0
    max_roll_pitch_rate_p95_abs_deg_s: float = 220.0
    max_yaw_rate_p95_abs_deg_s: float = 90.0
    max_abs_tilt_deg: float = 35.0
    min_zrange_mm: float = 180.0
    min_state_height_m: float = 0.20
    max_state_height_m: float = 1.20
    max_speed_m_s: float = 3.0
    min_source_horizontal_mm: float = 80.0


def evaluate_direct_raw_replay(rows: Iterable[dict[str, float]], thresholds: DirectRawGateThresholds) -> dict:
    parsed = list(rows)
    safe = [row for row in parsed if safe_row(row, thresholds)]
    close = [row for row in safe if horizontal_min_mm(row) < thresholds.close_range_mm]
    source = source_metrics(parsed, thresholds)
    failures = source_failures(source) if thresholds.require_source_health else []
    failures.extend(command_transform_failures(parsed, thresholds))
    if len(safe) < thresholds.min_safe_rows:
        failures.append("too_few_safe_rows")
    safe_metrics = metrics(safe)
    close_metrics = metrics(close)
    failures.extend(metric_failures("safe", safe_metrics, thresholds, close=False))
    if close:
        failures.extend(metric_failures("close", close_metrics, thresholds, close=True))
    return {
        "passed": not failures,
        "failures": failures,
        "rows": len(parsed),
        "safe_rows": len(safe),
        "close_safe_rows": len(close),
        "source": source,
        "source_tumble_rows": source["tumble_rows"],
        "safe": safe_metrics,
        "close": close_metrics,
        "thresholds": asdict(thresholds),
    }


def metric_failures(prefix: str, data: dict, thresholds: DirectRawGateThresholds, *, close: bool) -> list[str]:
    if not data:
        return []
    failures = []
    max_sat = thresholds.max_close_action_saturation if close else thresholds.max_safe_action_saturation
    if data["action_saturation_fraction"] > max_sat:
        failures.append(f"{prefix}_action_saturation")
    if data["thrust_percent_p05"] < thresholds.min_thrust_p05_percent:
        failures.append(f"{prefix}_low_thrust_p05")
    if data["thrust_percent_p95"] > thresholds.max_thrust_p95_percent:
        failures.append(f"{prefix}_high_thrust_p95")
    if max(data["roll_rate_abs_p95"], data["pitch_rate_abs_p95"]) > thresholds.max_roll_pitch_rate_p95_abs_deg_s:
        failures.append(f"{prefix}_roll_pitch_rate_p95")
    if data["yaw_rate_abs_p95"] > thresholds.max_yaw_rate_p95_abs_deg_s:
        failures.append(f"{prefix}_yaw_rate_p95")
    return failures


def metrics(rows: list[dict[str, float]]) -> dict:
    if not rows:
        return {}
    actions = np.asarray([[value(row, col) for col in ACTION_COLUMNS] for row in rows], dtype=np.float32)
    thrust = np.asarray([value(row, "thrust_percent") for row in rows], dtype=np.float32)
    rates = {col: np.asarray([value(row, col) for row in rows], dtype=np.float32) for col in RATE_COLUMNS}
    hmin = np.asarray([horizontal_min_mm(row) for row in rows], dtype=np.float32)
    return {
        "action_saturation_fraction": float(np.mean(np.abs(actions) >= 0.999)),
        "action_abs_p95": float(np.quantile(np.abs(actions), 0.95)),
        "thrust_percent_p05": float(np.quantile(thrust, 0.05)),
        "thrust_percent_median": float(np.median(thrust)),
        "thrust_percent_p95": float(np.quantile(thrust, 0.95)),
        "roll_rate_abs_p95": float(np.quantile(np.abs(rates["roll_rate_deg_s"]), 0.95)),
        "pitch_rate_abs_p95": float(np.quantile(np.abs(rates["pitch_rate_deg_s"]), 0.95)),
        "yaw_rate_abs_p95": float(np.quantile(np.abs(rates["yaw_rate_deg_s"]), 0.95)),
        "horizontal_min_mm": float(np.nanmin(hmin)),
    }


def source_metrics(rows: list[dict[str, float]], thresholds: DirectRawGateThresholds) -> dict:
    if not rows:
        return {
            "tumble_rows": 0,
            "canfly_failed_rows": 0,
            "near_contact_rows": 0,
            "high_speed_rows": 0,
            "extreme_tilt_rows": 0,
            "precontact_high_speed_rows": 0,
            "horizontal_min_mm": 4000.0,
            "speed_max_m_s": 0.0,
            "precontact_horizontal_speed_max_m_s": 0.0,
            "tilt_max_abs_deg": 0.0,
        }
    hmins = np.asarray([horizontal_min_mm(row) for row in rows], dtype=np.float32)
    speeds = np.asarray([speed_m_s(row) for row in rows], dtype=np.float32)
    horizontal_speeds = np.asarray([horizontal_speed_m_s(row) for row in rows], dtype=np.float32)
    tilts = np.asarray([tilt_abs_deg(row) for row in rows], dtype=np.float32)
    precontact = np.asarray([precontact_row(row, thresholds) for row in rows], dtype=bool)
    precontact_speeds = horizontal_speeds[precontact]
    return {
        "tumble_rows": int(sum(value(row, "sys.isTumbled") > 0.0 for row in rows)),
        "canfly_failed_rows": int(sum(value(row, "sys.canfly", 1.0) <= 0.0 for row in rows)),
        "near_contact_rows": int(np.sum(hmins < thresholds.min_source_horizontal_mm)),
        "high_speed_rows": int(np.sum(speeds > thresholds.max_speed_m_s)),
        "extreme_tilt_rows": int(np.sum(tilts > thresholds.max_abs_tilt_deg)),
        "precontact_high_speed_rows": int(np.sum(precontact_speeds > thresholds.max_precontact_horizontal_speed_m_s)),
        "horizontal_min_mm": float(np.nanmin(hmins)),
        "speed_max_m_s": float(np.nanmax(speeds)),
        "precontact_horizontal_speed_max_m_s": float(np.nanmax(precontact_speeds)) if precontact_speeds.size else 0.0,
        "tilt_max_abs_deg": float(np.nanmax(tilts)),
    }


def source_failures(source: dict) -> list[str]:
    failures = []
    for key, failure in (
        ("tumble_rows", "source_tumble"),
        ("canfly_failed_rows", "source_canfly_failed"),
        ("near_contact_rows", "source_near_contact"),
        ("high_speed_rows", "source_high_speed"),
        ("extreme_tilt_rows", "source_extreme_tilt"),
        ("precontact_high_speed_rows", "source_precontact_drift"),
    ):
        if source[key] > 0:
            failures.append(failure)
    return failures


def command_transform_failures(rows: list[dict[str, float]], thresholds: DirectRawGateThresholds) -> list[str]:
    if not rows or not thresholds.require_commander_pitch_sign:
        return []
    if not any("commander_pitch_rate_deg_s" in row for row in rows):
        return ["missing_commander_pitch_sign"]
    mismatches = [
        row
        for row in rows
        if abs(value(row, "commander_pitch_rate_deg_s") + value(row, "pitch_rate_deg_s")) > thresholds.pitch_sign_tolerance_deg_s
    ]
    return ["commander_pitch_sign_mismatch"] if mismatches else []


def safe_row(row: dict[str, float], thresholds: DirectRawGateThresholds) -> bool:
    if value(row, "sys.isTumbled") > 0.0 or value(row, "sys.canfly", 1.0) <= 0.0:
        return False
    if abs(value(row, "stabilizer.roll")) > thresholds.max_abs_tilt_deg:
        return False
    if abs(value(row, "stabilizer.pitch")) > thresholds.max_abs_tilt_deg:
        return False
    z = value(row, "stateEstimate.z", 0.5)
    if z < thresholds.min_state_height_m or z > thresholds.max_state_height_m:
        return False
    if speed_m_s(row) > thresholds.max_speed_m_s:
        return False
    return live_range_mm(row, "range.zrange") >= thresholds.min_zrange_mm


def precontact_row(row: dict[str, float], thresholds: DirectRawGateThresholds) -> bool:
    zrange = raw_range_mm(row, "range.zrange")
    if not np.isfinite(zrange) or zrange < thresholds.min_zrange_mm:
        return False
    z = value(row, "stateEstimate.z", 0.5)
    if z < thresholds.min_state_height_m or z > thresholds.max_state_height_m:
        return False
    return horizontal_min_mm(row) >= thresholds.precontact_min_horizontal_mm


def horizontal_speed_m_s(row: dict[str, float]) -> float:
    return float(np.linalg.norm([value(row, "stateEstimate.vx"), value(row, "stateEstimate.vy")]))


def speed_m_s(row: dict[str, float]) -> float:
    return float(np.linalg.norm([value(row, "stateEstimate.vx"), value(row, "stateEstimate.vy"), value(row, "stateEstimate.vz")]))


def tilt_abs_deg(row: dict[str, float]) -> float:
    values = (
        value(row, "stabilizer.roll"),
        value(row, "stabilizer.pitch"),
        value(row, "stateEstimate.roll"),
        value(row, "stateEstimate.pitch"),
    )
    finite = [abs(item) for item in values if np.isfinite(item)]
    return max(finite) if finite else 0.0


def horizontal_min_mm(row: dict[str, float]) -> float:
    values = [live_range_mm(row, key) for key in HORIZONTAL_RANGES]
    finite = [item for item in values if item < 32000.0]
    return min(finite) if finite else 4000.0


def live_range_mm(row: dict[str, float], key: str) -> float:
    raw = raw_range_mm(row, key)
    return 4000.0 if not np.isfinite(raw) else raw


def raw_range_mm(row: dict[str, float], key: str) -> float:
    raw = value(row, key, 4000.0)
    if raw <= 0.0 or not np.isfinite(raw):
        return float("nan")
    return 4000.0 if raw >= 32000.0 else raw


def value(row: dict[str, float], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default
