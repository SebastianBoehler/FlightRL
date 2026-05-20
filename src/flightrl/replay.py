from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


RANGE_KEYS = ("range.front", "range.back", "range.left", "range.right", "range.up", "range.zrange")
COMMAND_KEYS = ("vx_m_s", "vy_m_s", "vz_m_s", "yawrate_deg_s", "action_thrust", "action_roll_rate", "action_pitch_rate", "action_yaw_rate")
STATE_KEYS = (
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.vz",
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
)
TIME_KEY = "host_time_s"
CALIBRATION_REQUIRED = (
    TIME_KEY,
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
    "yawrate_deg_s",
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
)


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> dict[str, float]:
    xyz = np.asarray([[value(row, "stateEstimate.x"), value(row, "stateEstimate.y"), value(row, "stateEstimate.z")] for row in rows], dtype=np.float32)
    summary: dict[str, float] = {"rows": float(len(rows))}
    if len(rows) == 0:
        return summary
    summary["duration_s"] = max(value(rows[-1], TIME_KEY) - value(rows[0], TIME_KEY), 0.0)
    if any(has_any(row, ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z")) for row in rows):
        summary["mean_z_m"] = float(np.mean(xyz[:, 2]))
        summary["path_length_m"] = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))) if len(rows) > 1 else 0.0
        summary["xy_span_m"] = float(np.linalg.norm(np.ptp(xyz[:, :2], axis=0)))
    for key in RANGE_KEYS:
        values = np.asarray([value(row, key) / 1000.0 for row in rows], dtype=np.float32)
        valid = values[(values > 0.02) & (values < 4.0)]
        summary[f"{key}.valid_ratio"] = float(len(valid) / len(values))
        if len(valid):
            summary[f"{key}.min_m"] = float(np.min(valid))
    for key in COMMAND_KEYS:
        values = np.asarray([value(row, key) for row in rows if key in row], dtype=np.float32)
        if len(values):
            summary[f"{key}.abs_mean"] = float(np.mean(np.abs(values)))
            summary[f"{key}.abs_max"] = float(np.max(np.abs(values)))
    return summary


def compare(real: dict[str, float], sim: dict[str, float]) -> dict[str, float]:
    return {f"{key}.sim_minus_real": sim[key] - real[key] for key in sorted(set(real) & set(sim)) if key != "rows"}


def aligned_compare(real_rows: list[dict[str, str]], sim_rows: list[dict[str, str]], requested: list[str] | None = None) -> dict:
    alignment = alignment_context(real_rows, sim_rows)
    if alignment is None:
        return {"signals": {}, "samples": 0, "overlap_duration_s": 0.0}
    real_t, sim_t, mask, duration = alignment
    signals = {
        key: aligned_signal_metrics(real_rows, sim_rows, real_t, sim_t, mask, key)
        for key in requested_or_common(real_rows, sim_rows, requested)
    }
    signals = {key: metrics for key, metrics in signals.items() if metrics["samples"] > 0}
    return {"signals": signals, "samples": int(np.sum(mask)), "overlap_duration_s": duration}


def fit_linear_calibration(real_rows: list[dict[str, str]], sim_rows: list[dict[str, str]], requested: list[str] | None = None) -> dict:
    alignment = alignment_context(real_rows, sim_rows)
    if alignment is None:
        return {"signals": {}, "samples": 0, "overlap_duration_s": 0.0}
    real_t, sim_t, mask, duration = alignment
    signals = {}
    for key in requested_or_common(real_rows, sim_rows, requested):
        arrays = aligned_signal_arrays(real_rows, sim_rows, real_t, sim_t, mask, key)
        if arrays is None:
            continue
        real_values, sim_values = arrays
        signals[key] = fit_signal(real_values, sim_values)
    return {"signals": signals, "samples": int(np.sum(mask)), "overlap_duration_s": duration}


def fit_signal(real_values: np.ndarray, sim_values: np.ndarray) -> dict[str, float]:
    design = np.column_stack([sim_values, np.ones_like(sim_values)])
    scale, bias = np.linalg.lstsq(design, real_values, rcond=None)[0]
    fitted = sim_values * scale + bias
    raw_error = sim_values - real_values
    fitted_error = fitted - real_values
    return {
        "samples": int(len(real_values)),
        "scale": float(scale),
        "bias": float(bias),
        "raw_rmse": rmse(raw_error),
        "fitted_rmse": rmse(fitted_error),
        "raw_bias": float(np.mean(raw_error)),
        "fitted_bias": float(np.mean(fitted_error)),
    }


def assess_log_quality(
    rows: list[dict[str, str]],
    *,
    required: tuple[str, ...] = CALIBRATION_REQUIRED,
    min_rows: int = 100,
    min_duration_s: float = 5.0,
    min_range_valid_ratio: float = 0.25,
) -> dict:
    columns = set(rows[0]) if rows else set()
    missing = [key for key in required if key not in columns]
    report = {
        "rows": len(rows),
        "duration_s": duration_s(rows),
        "sample_rate_hz": sample_rate_hz(rows),
        "time_monotonic": time_monotonic(rows),
        "missing_columns": missing,
        "range_valid_ratio": range_valid_ratios(rows),
    }
    failures = []
    if report["rows"] < min_rows:
        failures.append("rows")
    if report["duration_s"] < min_duration_s:
        failures.append("duration")
    if not report["time_monotonic"]:
        failures.append("time_monotonic")
    if missing:
        failures.append("missing_columns")
    weak_ranges = [
        key
        for key in RANGE_KEYS
        if key in required and report["range_valid_ratio"].get(key, 0.0) < min_range_valid_ratio
    ]
    if weak_ranges:
        failures.append("range_validity")
    report["weak_range_columns"] = weak_ranges
    report["calibration_ready"] = not failures
    report["failures"] = failures
    return report


def duration_s(rows: list[dict[str, str]]) -> float:
    if len(rows) < 2:
        return 0.0
    return max(value(rows[-1], TIME_KEY) - value(rows[0], TIME_KEY), 0.0)


def sample_rate_hz(rows: list[dict[str, str]]) -> float:
    duration = duration_s(rows)
    return float((len(rows) - 1) / duration) if len(rows) > 1 and duration > 0 else 0.0


def time_monotonic(rows: list[dict[str, str]]) -> bool:
    if len(rows) < 2:
        return bool(rows)
    times = [value(row, TIME_KEY) for row in rows]
    return all(curr > prev for prev, curr in zip(times, times[1:]))


def range_valid_ratios(rows: list[dict[str, str]]) -> dict[str, float]:
    ratios = {}
    for key in RANGE_KEYS:
        if not rows or key not in rows[0]:
            continue
        values = np.asarray([value(row, key) for row in rows], dtype=np.float32)
        ratios[key] = float(np.mean((values > 20.0) & (values < 4000.0)))
    return ratios


def aligned_signal_metrics(real_rows, sim_rows, real_t: np.ndarray, sim_t: np.ndarray, mask: np.ndarray, key: str) -> dict[str, float]:
    arrays = aligned_signal_arrays(real_rows, sim_rows, real_t, sim_t, mask, key)
    if arrays is None:
        return {"samples": 0}
    real_aligned, sim_aligned = arrays
    error = sim_aligned - real_aligned
    return {
        "samples": int(len(error)),
        "rmse": rmse(error),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "real_mean": float(np.mean(real_aligned)),
        "sim_mean": float(np.mean(sim_aligned)),
    }


def aligned_signal_arrays(real_rows, sim_rows, real_t: np.ndarray, sim_t: np.ndarray, mask: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray] | None:
    real_values = np.asarray([value(row, key) for row in real_rows], dtype=np.float32)
    sim_values = np.asarray([value(row, key) for row in sim_rows], dtype=np.float32)
    real_aligned = real_values[mask]
    sim_aligned = np.interp(real_t[mask], sim_t, sim_values)
    finite = np.isfinite(real_aligned) & np.isfinite(sim_aligned) & valid_signal_mask(key, real_aligned, sim_aligned)
    if not np.any(finite):
        return None
    return real_aligned[finite], sim_aligned[finite]


def alignment_context(real_rows: list[dict[str, str]], sim_rows: list[dict[str, str]]):
    if not real_rows or not sim_rows:
        return None
    real_t = relative_time(real_rows)
    sim_t = relative_time(sim_rows)
    if len(real_t) < 2 or len(sim_t) < 2:
        return None
    start = max(real_t[0], sim_t[0])
    end = min(real_t[-1], sim_t[-1])
    mask = (real_t >= start) & (real_t <= end)
    if end <= start or not np.any(mask):
        return None
    return real_t, sim_t, mask, float(end - start)


def requested_or_common(real_rows: list[dict[str, str]], sim_rows: list[dict[str, str]], requested: list[str] | None) -> list[str]:
    if requested:
        return [key for key in requested if key in real_rows[0] and key in sim_rows[0]]
    common = set(real_rows[0]) & set(sim_rows[0])
    return [key for key in (*STATE_KEYS, *RANGE_KEYS, *COMMAND_KEYS) if key in common]


def valid_signal_mask(key: str, real_values: np.ndarray, sim_values: np.ndarray) -> np.ndarray:
    if key.startswith("range."):
        return (real_values > 20.0) & (real_values < 4000.0) & (sim_values > 20.0) & (sim_values < 4000.0)
    return np.ones(real_values.shape, dtype=bool)


def relative_time(rows: list[dict[str, str]]) -> np.ndarray:
    values = np.asarray([value(row, TIME_KEY) for row in rows], dtype=np.float64)
    return values - values[0]


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error * error)))


def has_any(row: dict[str, str], keys: tuple[str, ...]) -> bool:
    return any(key in row for key in keys)


def value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
