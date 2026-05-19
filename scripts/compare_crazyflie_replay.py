from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


RANGE_KEYS = ("range.front", "range.back", "range.left", "range.right", "range.up", "range.zrange")
COMMAND_KEYS = ("vx_m_s", "vy_m_s", "vz_m_s", "yawrate_deg_s", "action_thrust", "action_roll_rate", "action_pitch_rate", "action_yaw_rate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and compare Crazyflie-style real/sim replay CSV files")
    parser.add_argument("--real", required=True)
    parser.add_argument("--sim", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    real_summary = summarize(load_rows(args.real))
    result = {"real": real_summary}
    if args.sim:
        sim_summary = summarize(load_rows(args.sim))
        result["sim"] = sim_summary
        result["delta"] = compare(real_summary, sim_summary)
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
    summary["duration_s"] = max(value(rows[-1], "host_time_s") - value(rows[0], "host_time_s"), 0.0)
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


def value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
