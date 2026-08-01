from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from flightrl.vertical_clearance import LIVE_VERTICAL_BOTTOM_CLEARANCE_M, LIVE_VERTICAL_TOP_CLEARANCE_M, vertical_velocity_from_clearance


@dataclass(frozen=True, slots=True)
class RangerReading:
    front_m: float
    back_m: float
    left_m: float
    right_m: float
    up_m: float
    zrange_m: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare logged vertical commands with the tuned clearance model.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-vertical-speed-m-s", type=float, default=0.08)
    parser.add_argument("--hard-clearance-m", type=float, default=0.06)
    args = parser.parse_args()

    report = build_report(args.input, args.max_vertical_speed_m_s, args.hard_clearance_m)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"vertical_clearance_report={output}")
    print(f"inputs={len(report['logs'])} samples={report['summary']['samples']}")


def build_report(paths: list[str], max_vertical_speed_m_s: float, hard_clearance_m: float) -> dict[str, Any]:
    logs = [summarize_log(Path(path), max_vertical_speed_m_s, hard_clearance_m) for path in paths]
    return {
        "logs": logs,
        "summary": combine(logs),
        "model": {
            "top_clearance_m": LIVE_VERTICAL_TOP_CLEARANCE_M,
            "bottom_clearance_m": LIVE_VERTICAL_BOTTOM_CLEARANCE_M,
            "hard_clearance_m": hard_clearance_m,
            "max_vertical_speed_m_s": max_vertical_speed_m_s,
        },
        "safety": "Replay-only vertical command comparison; no live hardware commands were produced.",
    }


def summarize_log(path: Path, max_vertical_speed_m_s: float, hard_clearance_m: float) -> dict[str, Any]:
    rows = load_rows(path)
    records = []
    for row in rows:
        reading = reading_from_telemetry(row)
        tuned_vz = vertical_velocity_from_clearance(
            reading,
            hard_clearance_m=hard_clearance_m,
            max_vertical_speed_m_s=max_vertical_speed_m_s,
        )
        records.append(
            {
                "logged_vz": value(row, "vz_m_s"),
                "tuned_vz": tuned_vz,
                "top_m": reading.up_m,
                "bottom_m": reading.zrange_m,
            }
        )
    return {"input": str(path), **metrics(records)}


def metrics(records: list[dict[str, float]]) -> dict[str, Any]:
    top = [record for record in records if record["top_m"] < 0.45]
    bottom_guard = [record for record in records if record["bottom_m"] < 0.30]
    squeezed = [record for record in records if record["top_m"] < 0.45 and record["bottom_m"] < 0.30]
    return {
        "samples": len(records),
        "top_lt_45cm": len(top),
        "bottom_lt_30cm": len(bottom_guard),
        "squeezed_top_bottom": len(squeezed),
        "logged_down_when_bottom_guard": count(bottom_guard, "logged_vz", lambda value: value < -1e-4),
        "tuned_down_when_bottom_guard": count(bottom_guard, "tuned_vz", lambda value: value < -1e-4),
        "logged_down_when_squeezed": count(squeezed, "logged_vz", lambda value: value < -1e-4),
        "tuned_down_when_squeezed": count(squeezed, "tuned_vz", lambda value: value < -1e-4),
        "logged_vz": describe([record["logged_vz"] for record in records]),
        "tuned_vz": describe([record["tuned_vz"] for record in records]),
    }


def combine(logs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "samples",
        "top_lt_45cm",
        "bottom_lt_30cm",
        "squeezed_top_bottom",
        "logged_down_when_bottom_guard",
        "tuned_down_when_bottom_guard",
        "logged_down_when_squeezed",
        "tuned_down_when_squeezed",
    )
    return {key: int(sum(log[key] for log in logs)) for key in keys}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Vertical Clearance Replay Report",
        "",
        f"- Samples: `{report['summary']['samples']}`",
        f"- Bottom-guard logged-down samples: `{report['summary']['logged_down_when_bottom_guard']}`",
        f"- Bottom-guard tuned-down samples: `{report['summary']['tuned_down_when_bottom_guard']}`",
        "",
        "| log | samples | top<45 | bottom<30 | squeezed | logged down bottom | tuned down bottom |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for log in report["logs"]:
        lines.append(
            f"| {Path(log['input']).name} | {log['samples']} | {log['top_lt_45cm']} | {log['bottom_lt_30cm']} | "
            f"{log['squeezed_top_bottom']} | {log['logged_down_when_bottom_guard']} | {log['tuned_down_when_bottom_guard']} |"
        )
    return "\n".join(lines)


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open() as handle:
        return [{key: parse_float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def parse_float(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def value(row: dict[str, float], key: str) -> float:
    raw = row.get(key, 0.0)
    return raw if math.isfinite(raw) else 0.0


def reading_from_telemetry(values: dict[str, float]) -> RangerReading:
    return RangerReading(*(_range_m(values, key) for key in (
        "range.front",
        "range.back",
        "range.left",
        "range.right",
        "range.up",
        "range.zrange",
    )))


def _range_m(values: dict[str, float], key: str) -> float:
    raw = value(values, key)
    if raw >= 32000.0:
        return 4.0
    return raw / 1000.0


def count(records: list[dict[str, float]], key: str, predicate) -> int:
    return sum(1 for record in records if predicate(record[key]))


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float32)
    return {"min": float(np.min(array)), "mean": float(np.mean(array)), "max": float(np.max(array))}


if __name__ == "__main__":
    main()
