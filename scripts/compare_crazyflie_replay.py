from __future__ import annotations

import argparse
import csv
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and compare Crazyflie-style real/sim replay CSV files")
    parser.add_argument("--real", required=True)
    parser.add_argument("--sim", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--align-time", action="store_true", help="resample sim signals onto real timestamps and report errors")
    parser.add_argument("--signals", nargs="*", default=None, help="optional signal list for --align-time")
    args = parser.parse_args()

    real_rows = load_rows(args.real)
    real_summary = summarize(real_rows)
    result = {"real": real_summary}
    if args.sim:
        sim_rows = load_rows(args.sim)
        sim_summary = summarize(sim_rows)
        result["sim"] = sim_summary
        result["delta"] = compare(real_summary, sim_summary)
        if args.align_time:
            result["aligned"] = aligned_compare(real_rows, sim_rows, args.signals)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        print(f"wrote {output}")
    else:
        print(text)


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
    keys = sorted(set(real) & set(sim))
    return {f"{key}.sim_minus_real": sim[key] - real[key] for key in keys if key != "rows"}


def aligned_compare(real_rows: list[dict[str, str]], sim_rows: list[dict[str, str]], requested: list[str] | None = None) -> dict:
    if not real_rows or not sim_rows:
        return {"signals": {}, "samples": 0, "overlap_duration_s": 0.0}
    real_t = relative_time(real_rows)
    sim_t = relative_time(sim_rows)
    if len(real_t) < 2 or len(sim_t) < 2:
        return {"signals": {}, "samples": 0, "overlap_duration_s": 0.0}
    start = max(real_t[0], sim_t[0])
    end = min(real_t[-1], sim_t[-1])
    mask = (real_t >= start) & (real_t <= end)
    if end <= start or not np.any(mask):
        return {"signals": {}, "samples": 0, "overlap_duration_s": 0.0}
    columns = requested or aligned_signal_candidates(real_rows, sim_rows)
    signals = {
        key: aligned_signal_metrics(real_rows, sim_rows, real_t, sim_t, mask, key)
        for key in columns
        if key in real_rows[0] and key in sim_rows[0]
    }
    signals = {key: metrics for key, metrics in signals.items() if metrics["samples"] > 0}
    return {"signals": signals, "samples": int(np.sum(mask)), "overlap_duration_s": float(end - start)}


def aligned_signal_candidates(real_rows: list[dict[str, str]], sim_rows: list[dict[str, str]]) -> list[str]:
    common = set(real_rows[0]) & set(sim_rows[0])
    preferred = [*STATE_KEYS, *RANGE_KEYS, *COMMAND_KEYS]
    return [key for key in preferred if key in common]


def aligned_signal_metrics(real_rows, sim_rows, real_t: np.ndarray, sim_t: np.ndarray, mask: np.ndarray, key: str) -> dict[str, float]:
    real_values = np.asarray([value(row, key) for row in real_rows], dtype=np.float32)
    sim_values = np.asarray([value(row, key) for row in sim_rows], dtype=np.float32)
    real_aligned = real_values[mask]
    sim_aligned = np.interp(real_t[mask], sim_t, sim_values)
    finite = np.isfinite(real_aligned) & np.isfinite(sim_aligned) & valid_signal_mask(key, real_aligned, sim_aligned)
    if not np.any(finite):
        return {"samples": 0}
    error = sim_aligned[finite] - real_aligned[finite]
    return {
        "samples": int(np.sum(finite)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "real_mean": float(np.mean(real_aligned[finite])),
        "sim_mean": float(np.mean(sim_aligned[finite])),
    }


def valid_signal_mask(key: str, real_values: np.ndarray, sim_values: np.ndarray) -> np.ndarray:
    if key.startswith("range."):
        return (real_values > 20.0) & (real_values < 4000.0) & (sim_values > 20.0) & (sim_values < 4000.0)
    return np.ones(real_values.shape, dtype=bool)


def relative_time(rows: list[dict[str, str]]) -> np.ndarray:
    values = np.asarray([value(row, TIME_KEY) for row in rows], dtype=np.float64)
    return values - values[0]


def has_any(row: dict[str, str], keys: tuple[str, ...]) -> bool:
    return any(key in row for key in keys)


def value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
