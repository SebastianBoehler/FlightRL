from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any

from .noise_stats import aggregate_signal_stats, read_rows, segment_summary


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
MIN_STATIONARY_ROWS = 30
MIN_STATIONARY_SAMPLE_RATE_HZ = 5.0
MIN_SIGNAL_SAMPLES = 30
MIN_SIGNAL_VALID_RATIO = 0.5


def summarize_stationary_noise(
    path: Path,
    *,
    columns: list[str] | None = None,
    min_duration_s: float = 30.0,
    max_position_span_m: float = 0.08,
    max_attitude_span_deg: float = 6.0,
    max_range_span_mm: float = 300.0,
) -> dict[str, Any]:
    validate_noise_thresholds(
        min_duration_s=min_duration_s,
        max_position_span_m=max_position_span_m,
        max_attitude_span_deg=max_attitude_span_deg,
        max_range_span_mm=max_range_span_mm,
    )
    return summarize_stationary_noise_logs(
        [path],
        columns=columns,
        min_duration_s=min_duration_s,
        max_position_span_m=max_position_span_m,
        max_attitude_span_deg=max_attitude_span_deg,
        max_range_span_mm=max_range_span_mm,
    )


def summarize_stationary_noise_logs(
    paths: list[Path],
    *,
    columns: list[str] | None = None,
    min_duration_s: float = 30.0,
    max_position_span_m: float = 0.08,
    max_attitude_span_deg: float = 6.0,
    max_range_span_mm: float = 300.0,
) -> dict[str, Any]:
    validate_noise_thresholds(
        min_duration_s=min_duration_s,
        max_position_span_m=max_position_span_m,
        max_attitude_span_deg=max_attitude_span_deg,
        max_range_span_mm=max_range_span_mm,
    )
    if not paths:
        raise ValueError("stationary-noise summary requires at least one input")
    selected_columns = columns or DEFAULT_COLUMNS
    if (
        not selected_columns
        or any(not isinstance(column, str) or not column for column in selected_columns)
        or len(set(selected_columns)) != len(selected_columns)
    ):
        raise ValueError("stationary-noise columns must be unique nonempty names")
    logs = [(path, read_rows(path)) for path in paths]
    stats = {column: aggregate_signal_stats(logs, column, max_range_span_mm=max_range_span_mm) for column in selected_columns}
    segments = [segment_summary(path, rows, selected_columns) for path, rows in logs]
    duration = sum(float(segment["duration_s"]) for segment in segments)
    rows_count = sum(len(rows) for _path, rows in logs)
    interval_count = sum(max(0, len(rows) - 1) for _path, rows in logs)
    sample_rate_hz = interval_count / duration if duration > 0 else 0.0
    position_span = max((float(segment["max_position_span_m"]) for segment in segments), default=0.0)
    attitude_span = max((float(segment["max_attitude_span_deg"]) for segment in segments), default=0.0)
    failures = []
    if duration < min_duration_s:
        failures.append("duration")
    if rows_count < MIN_STATIONARY_ROWS:
        failures.append("rows")
    if sample_rate_hz < MIN_STATIONARY_SAMPLE_RATE_HZ:
        failures.append("sample_rate")
    if position_span > max_position_span_m:
        failures.append("position_motion")
    if attitude_span > max_attitude_span_deg:
        failures.append("attitude_motion")
    missing = [column for column, values in stats.items() if values["samples"] == 0]
    if missing:
        failures.append("missing_columns")
    sparse = [column for column, values in stats.items() if int(values["samples"]) < MIN_SIGNAL_SAMPLES]
    if sparse:
        failures.append("signal_samples")
    low_valid_ratio = [
        column
        for column, values in stats.items()
        if values.get("valid_ratio") is None or float(values["valid_ratio"]) < MIN_SIGNAL_VALID_RATIO
    ]
    if low_valid_ratio:
        failures.append("signal_valid_ratio")
    if (
        any(int(values.get("nonfinite_samples", 0)) > 0 for values in stats.values())
        or any(int(segment["invalid_time_samples"]) > 0 for segment in segments)
    ):
        failures.append("nonfinite_values")
    if any(not bool(segment["time_monotonic"]) for segment in segments):
        failures.append("time_monotonic")
    return {
        "input": str(paths[0]) if len(paths) == 1 else f"{len(paths)} inputs",
        "inputs": [str(path) for path in paths],
        "segments": segments,
        "summary": {
            "rows": rows_count,
            "inputs": len(paths),
            "duration_s": duration,
            "sample_rate_hz": sample_rate_hz,
            "stationary_noise_ready": not failures,
            "failures": failures,
            "missing_columns": missing,
            "sparse_columns": sparse,
            "low_valid_ratio_columns": low_valid_ratio,
            "min_rows": MIN_STATIONARY_ROWS,
            "min_sample_rate_hz": MIN_STATIONARY_SAMPLE_RATE_HZ,
            "min_signal_samples": MIN_SIGNAL_SAMPLES,
            "min_signal_valid_ratio": MIN_SIGNAL_VALID_RATIO,
            "max_position_span_m": position_span,
            "max_attitude_span_deg": attitude_span,
            "max_range_span_mm": max_range_span_mm,
        },
        "signals": stats,
        "safety": "Stationary noise summary only; do not use as a deployment gate without replay validation.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Stationary Noise Summary",
        "",
        f"- Input: `{report['input']}`",
        f"- Inputs: `{summary.get('inputs', 1)}`",
        f"- Ready: `{summary['stationary_noise_ready']}`",
        f"- Failures: `{', '.join(summary['failures']) or 'none'}`",
        f"- Duration s: `{summary['duration_s']:.3f}`",
        f"- Sample rate Hz: `{summary['sample_rate_hz']:.3f}`",
        "",
        "| signal | samples | raw | invalid | excluded | valid ratio | std | span |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for signal, stats in report["signals"].items():
        lines.append(
            f"| {signal} | {stats['samples']} | {stats.get('raw_samples', stats['samples'])} | "
            f"{stats.get('invalid_samples', 0)} | {stats.get('excluded_samples', 0)} | {_fmt(stats.get('valid_ratio'))} | "
            f"{_fmt(stats['std'])} | {_fmt(stats['span'])} |"
        )
    if len(report.get("segments", [])) > 1:
        lines.extend(["", "## Segments", "", "| input | rows | duration s | position span m | attitude span deg |", "| --- | ---: | ---: | ---: | ---: |"])
        for segment in report["segments"]:
            lines.append(
                f"| {segment['input']} | {segment['rows']} | {segment['duration_s']:.3f} | "
                f"{segment['max_position_span_m']:.6g} | {segment['max_attitude_span_deg']:.6g} |"
            )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def _fmt(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def validate_noise_thresholds(
    *,
    min_duration_s: float,
    max_position_span_m: float,
    max_attitude_span_deg: float,
    max_range_span_mm: float,
) -> None:
    values = (
        min_duration_s,
        max_position_span_m,
        max_attitude_span_deg,
        max_range_span_mm,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or value < 0.0
        for value in values
    ):
        raise ValueError("stationary-noise thresholds must be finite and nonnegative")
