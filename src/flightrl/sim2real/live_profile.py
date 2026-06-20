from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from flightrl.sixdof.sensor_model import SixDofSensorProfile


RANGE_COLUMNS = ("range.front", "range.back", "range.left", "range.right", "range.up", "range.zrange")
STATE_COLUMNS = ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z")
VELOCITY_COLUMNS = ("stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz")
GYRO_COLUMNS = ("gyro.x", "gyro.y", "gyro.z")


def build_live_sim_profile(
    *,
    flight_logs: list[Path],
    stationary_logs: list[Path],
    latency_report: Path | None = None,
    name: str = "rich_live",
) -> dict[str, Any]:
    flight_rows = [row for path in flight_logs for row in read_rows(path)]
    stationary_rows = [row for path in stationary_logs for row in read_rows(path)]
    stable_flight = [row for row in flight_rows if stable_flight_row(row)]
    noise_rows = stationary_rows or stable_flight
    sample_period_s = median_step_s(flight_rows)
    latency = latency_s(latency_report) if latency_report else None
    profile = SixDofSensorProfile(
        name=name,
        state_noise_std_m=bounded(max_robust_diff_std(noise_rows, STATE_COLUMNS), 0.0, 0.08),
        velocity_noise_std_m_s=bounded(max_robust_diff_std(noise_rows, VELOCITY_COLUMNS), 0.0, 1.0),
        body_rate_noise_std_rad_s=bounded(np.deg2rad(max_signal_std(noise_rows, GYRO_COLUMNS)), 0.0, 8.0),
        range_noise_std_m=bounded(max_robust_range_noise(noise_rows), 0.002, 0.12),
        range_dropout_prob=bounded(range_dropout_probability(flight_rows + stationary_rows), 0.0, 0.20),
        action_lag_s=bounded(latency if latency is not None else sample_period_s * 2.0, 0.0, 0.15),
    )
    return {
        "inputs": {
            "flight_logs": [str(path) for path in flight_logs],
            "stationary_logs": [str(path) for path in stationary_logs],
            "latency_report": str(latency_report) if latency_report else None,
        },
        "summary": {
            "flight_rows": len(flight_rows),
            "stable_flight_rows": len(stable_flight),
            "stationary_rows": len(stationary_rows),
            "sample_period_s": sample_period_s,
            "latency_source": "latency_report" if latency is not None else "sample_period_x2",
            "battery_v": quantiles(values(flight_rows, "pm.vbat")),
            "hmin_m": quantiles(horizontal_min_values(stable_flight)),
            "tumbled_rows": sum(1 for row in flight_rows if truthy(row.get("sys.isTumbled"))),
        },
        "sensor_profile": profile.as_report(),
        "safety": "Offline simulator profile only; direct live control still requires replay, shadow, and hardware approval gates.",
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def stable_flight_row(row: dict[str, str]) -> bool:
    if truthy(row.get("sys.isTumbled")):
        return False
    flying = as_float(row.get("sys.isFlying"))
    if flying is not None and flying < 0.5:
        return False
    roll = abs(as_float(row.get("stateEstimate.roll")) or as_float(row.get("stabilizer.roll")) or 0.0)
    pitch = abs(as_float(row.get("stateEstimate.pitch")) or as_float(row.get("stabilizer.pitch")) or 0.0)
    return roll < 35.0 and pitch < 35.0


def median_step_s(rows: list[dict[str, str]]) -> float:
    times = sorted(value for row in rows if (value := as_float(row.get("host_time_s"))) is not None)
    steps = [b - a for a, b in zip(times, times[1:], strict=False) if 0.0 < b - a < 0.2]
    return float(median(steps)) if steps else 0.01


def latency_s(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    if not summary.get("latency_ready"):
        return None
    value = summary.get("median_latency_s")
    return float(value) if value is not None else None


def max_robust_range_noise(rows: list[dict[str, str]]) -> float:
    return max((robust_diff_std(range_values(rows, column)) for column in RANGE_COLUMNS), default=0.0)


def max_robust_diff_std(rows: list[dict[str, str]], columns: tuple[str, ...]) -> float:
    return max((robust_diff_std(values(rows, column)) for column in columns), default=0.0)


def max_signal_std(rows: list[dict[str, str]], columns: tuple[str, ...]) -> float:
    return max((std(values(rows, column)) for column in columns), default=0.0)


def robust_diff_std(series: list[float]) -> float:
    if len(series) < 3:
        return 0.0
    diffs = np.diff(np.asarray(series, dtype=np.float32))
    centered = np.abs(diffs - np.median(diffs))
    return float(1.4826 * np.median(centered) / np.sqrt(2.0))


def range_values(rows: list[dict[str, str]], column: str) -> list[float]:
    output = []
    for row in rows:
        value = as_float(row.get(column))
        if value is None or value >= 32000.0:
            continue
        output.append(value / 1000.0)
    return output


def horizontal_min_values(rows: list[dict[str, str]]) -> list[float]:
    direct = values(rows, "min_horizontal_range_m")
    if direct:
        return direct
    output = []
    for row in rows:
        readings = [value for column in RANGE_COLUMNS[:4] if (value := as_float(row.get(column))) is not None and value < 32000.0]
        if readings:
            output.append(min(readings) / 1000.0)
    return output


def range_dropout_probability(rows: list[dict[str, str]]) -> float:
    total = 0
    missing = 0
    for column in RANGE_COLUMNS:
        values = [as_float(row.get(column)) for row in rows]
        for previous, current, following in zip(values, values[1:], values[2:], strict=False):
            if is_finite_range(previous) and is_finite_range(following):
                total += 1
                missing += int(not is_finite_range(current))
    return float(missing / total) if total else 0.0


def is_finite_range(value: float | None) -> bool:
    return value is not None and value < 32000.0


def values(rows: list[dict[str, str]], column: str) -> list[float]:
    return [value for row in rows if (value := as_float(row.get(column))) is not None]


def quantiles(series: list[float]) -> dict[str, float | None]:
    if not series:
        return {"min": None, "p05": None, "median": None, "p95": None, "max": None}
    array = np.asarray(series, dtype=np.float32)
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def std(series: list[float]) -> float:
    return float(np.std(np.asarray(series, dtype=np.float32))) if series else 0.0


def bounded(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def truthy(value: object) -> bool:
    number = as_float(value)
    return bool(number and number > 0.5)


def as_float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def render_markdown(report: dict[str, Any]) -> str:
    profile = report["sensor_profile"]
    summary = report["summary"]
    return "\n".join(
        [
            "# Rich Live Simulator Profile",
            "",
            f"- Flight rows: `{summary['flight_rows']}`",
            f"- Stable flight rows: `{summary['stable_flight_rows']}`",
            f"- Stationary rows: `{summary['stationary_rows']}`",
            f"- Sample period s: `{summary['sample_period_s']:.6f}`",
            f"- Latency source: `{summary['latency_source']}`",
            "",
            "## Sensor Profile",
            "",
            "```json",
            json.dumps(profile, indent=2, sort_keys=True),
            "```",
            "",
            report["safety"],
        ]
    )


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
