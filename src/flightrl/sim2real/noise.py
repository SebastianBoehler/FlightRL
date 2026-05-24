from __future__ import annotations

import csv
import json
from math import sqrt
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS = [
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "acc.x",
    "acc.y",
    "acc.z",
    "gyro.x",
    "gyro.y",
    "gyro.z",
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "range.zrange",
]


def summarize_stationary_noise(
    path: Path,
    *,
    columns: list[str] | None = None,
    min_duration_s: float = 30.0,
    max_position_span_m: float = 0.08,
    max_attitude_span_deg: float = 6.0,
) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open()))
    stats = {
        column: describe([value for row in rows if (value := _as_float(row.get(column))) is not None])
        for column in (columns or DEFAULT_COLUMNS)
    }
    duration = duration_s(rows)
    failures = []
    if duration < min_duration_s:
        failures.append("duration")
    if max_span(stats, ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z"]) > max_position_span_m:
        failures.append("position_motion")
    if max_span(stats, ["stabilizer.roll", "stabilizer.pitch"]) > max_attitude_span_deg:
        failures.append("attitude_motion")
    missing = [column for column, values in stats.items() if values["samples"] == 0]
    if missing:
        failures.append("missing_columns")
    return {
        "input": str(path),
        "summary": {
            "rows": len(rows),
            "duration_s": duration,
            "sample_rate_hz": (len(rows) - 1) / duration if duration > 0 and len(rows) > 1 else 0.0,
            "stationary_noise_ready": not failures,
            "failures": failures,
            "missing_columns": missing,
            "max_position_span_m": max_span(stats, ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z"]),
            "max_attitude_span_deg": max_span(stats, ["stabilizer.roll", "stabilizer.pitch"]),
        },
        "signals": stats,
        "safety": "Stationary noise summary only; do not use as a deployment gate without replay validation.",
    }


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "mean": None, "std": None, "min": None, "max": None, "span": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "samples": len(values),
        "mean": mean,
        "std": sqrt(variance),
        "min": min(values),
        "max": max(values),
        "span": max(values) - min(values),
    }


def duration_s(rows: list[dict[str, str]]) -> float:
    times = [_as_float(row.get("host_time_s")) for row in rows]
    valid = [value for value in times if value is not None]
    return max(valid) - min(valid) if len(valid) >= 2 else 0.0


def max_span(stats: dict[str, dict[str, Any]], columns: list[str]) -> float:
    spans = [stats.get(column, {}).get("span") for column in columns]
    return max([float(span) for span in spans if span is not None], default=0.0)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Stationary Noise Summary",
        "",
        f"- Input: `{report['input']}`",
        f"- Ready: `{summary['stationary_noise_ready']}`",
        f"- Failures: `{', '.join(summary['failures']) or 'none'}`",
        f"- Duration s: `{summary['duration_s']:.3f}`",
        f"- Sample rate Hz: `{summary['sample_rate_hz']:.3f}`",
        "",
        "| signal | samples | std | span |",
        "| --- | ---: | ---: | ---: |",
    ]
    for signal, stats in report["signals"].items():
        lines.append(f"| {signal} | {stats['samples']} | {_fmt(stats['std'])} | {_fmt(stats['span'])} |")
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def _fmt(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def _as_float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
