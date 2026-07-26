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
RANGE_COLUMNS = {column for column in DEFAULT_COLUMNS if column.startswith("range.")}
RANGE_NO_RETURN_MM = 32000.0


def summarize_stationary_noise(
    path: Path,
    *,
    columns: list[str] | None = None,
    min_duration_s: float = 30.0,
    max_position_span_m: float = 0.08,
    max_attitude_span_deg: float = 6.0,
    max_range_span_mm: float = 300.0,
) -> dict[str, Any]:
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
    selected_columns = columns or DEFAULT_COLUMNS
    logs = [(path, read_rows(path)) for path in paths]
    all_rows = [row for _path, rows in logs for row in rows]
    stats = {column: aggregate_signal_stats(logs, column, max_range_span_mm=max_range_span_mm) for column in selected_columns}
    segments = [segment_summary(path, rows, selected_columns) for path, rows in logs]
    duration = sum(float(segment["duration_s"]) for segment in segments)
    rows_count = sum(len(rows) for _path, rows in logs)
    interval_count = sum(max(0, len(rows) - 1) for _path, rows in logs)
    position_span = max((float(segment["max_position_span_m"]) for segment in segments), default=0.0)
    attitude_span = max((float(segment["max_attitude_span_deg"]) for segment in segments), default=0.0)
    failures = []
    if duration < min_duration_s:
        failures.append("duration")
    if position_span > max_position_span_m:
        failures.append("position_motion")
    if attitude_span > max_attitude_span_deg:
        failures.append("attitude_motion")
    missing = [column for column, values in stats.items() if values["raw_samples"] == 0]
    if missing:
        failures.append("missing_columns")
    return {
        "input": str(paths[0]) if len(paths) == 1 else f"{len(paths)} inputs",
        "inputs": [str(path) for path in paths],
        "segments": segments,
        "summary": {
            "rows": rows_count,
            "inputs": len(paths),
            "duration_s": duration,
            "sample_rate_hz": interval_count / duration if duration > 0 else 0.0,
            "stationary_noise_ready": not failures,
            "failures": failures,
            "missing_columns": missing,
            "max_position_span_m": position_span,
            "max_attitude_span_deg": attitude_span,
            "max_range_span_mm": max_range_span_mm,
        },
        "signals": stats,
        "safety": "Stationary noise summary only; do not use as a deployment gate without replay validation.",
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def segment_summary(path: Path, rows: list[dict[str, str]], columns: list[str]) -> dict[str, Any]:
    stats = {column: signal_stats(rows, column) for column in columns}
    duration = duration_s(rows)
    return {
        "input": str(path),
        "rows": len(rows),
        "duration_s": duration,
        "sample_rate_hz": (len(rows) - 1) / duration if duration > 0 and len(rows) > 1 else 0.0,
        "max_position_span_m": max_span(stats, ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z"]),
        "max_attitude_span_deg": max_span(stats, ["stabilizer.roll", "stabilizer.pitch"]),
    }


def aggregate_signal_stats(
    logs: list[tuple[Path, list[dict[str, str]]]],
    column: str,
    *,
    max_range_span_mm: float,
) -> dict[str, float | int | None | list[str]]:
    segments = [(path, rows, signal_stats(rows, column)) for path, rows in logs]
    included = [(path, rows, segment) for path, rows, segment in segments if include_segment_for_noise(column, segment, max_range_span_mm)]
    excluded = [(path, segment) for path, _rows, segment in segments if not include_segment_for_noise(column, segment, max_range_span_mm)]
    all_values = [value for _path, rows, _segment in included for row in rows if (value := valid_column_float(row, column)) is not None]
    stats = describe(all_values)
    valid_samples = sum(int(segment["samples"]) for _path, _rows, segment in included)
    raw_samples = sum(int(segment["raw_samples"]) for _path, _rows, segment in segments)
    invalid_samples = sum(int(segment["invalid_samples"]) for _path, _rows, segment in segments)
    excluded_samples = sum(int(segment["samples"]) for _path, segment in excluded)
    if valid_samples:
        stats["std"] = pooled_std([segment for _path, _rows, segment in included])
        stats["global_span"] = stats["span"]
        stats["span"] = max((float(segment["span"]) for _path, _rows, segment in included if segment["span"] is not None), default=0.0)
    stats["raw_samples"] = raw_samples
    stats["invalid_samples"] = invalid_samples
    stats["excluded_samples"] = excluded_samples
    stats["excluded_segments"] = len(excluded)
    stats["excluded_inputs"] = [str(path) for path, _segment in excluded]
    stats["valid_ratio"] = valid_samples / raw_samples if raw_samples else None
    return stats


def include_segment_for_noise(column: str, segment: dict[str, float | int | None], max_range_span_mm: float) -> bool:
    if column not in RANGE_COLUMNS or int(segment["samples"]) <= 0:
        return True
    span = segment["span"]
    return span is None or float(span) <= max_range_span_mm


def signal_stats(rows: list[dict[str, str]], column: str) -> dict[str, float | int | None]:
    raw_values = [value for row in rows if (value := _as_float(row.get(column))) is not None]
    valid_values = [value for value in raw_values if valid_signal_value(column, value)]
    stats = describe(valid_values)
    stats["raw_samples"] = len(raw_values)
    stats["invalid_samples"] = len(raw_values) - len(valid_values)
    stats["valid_ratio"] = len(valid_values) / len(raw_values) if raw_values else None
    return stats


def valid_column_float(row: dict[str, str], column: str) -> float | None:
    value = _as_float(row.get(column))
    if value is None or not valid_signal_value(column, value):
        return None
    return value


def valid_signal_value(column: str, value: float) -> bool:
    if column not in RANGE_COLUMNS:
        return True
    return 0.0 < value < RANGE_NO_RETURN_MM


def pooled_std(segments: list[dict[str, float | int | None]]) -> float:
    total = sum(int(segment["samples"]) for segment in segments)
    if total <= 0:
        return 0.0
    variance = sum(int(segment["samples"]) * float(segment["std"] or 0.0) ** 2 for segment in segments) / total
    return sqrt(variance)


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


def _as_float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
