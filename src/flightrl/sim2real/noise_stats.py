from __future__ import annotations

import csv
from math import isfinite, sqrt
from pathlib import Path
from typing import Any


RANGE_COLUMNS = {
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "range.zrange",
}
RANGE_NO_RETURN_MM = 32000.0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def segment_summary(
    path: Path,
    rows: list[dict[str, str]],
    columns: list[str],
) -> dict[str, Any]:
    stats = {column: signal_stats(rows, column) for column in columns}
    time_stats = signal_stats(rows, "host_time_s")
    times = [_as_float(row.get("host_time_s")) for row in rows]
    finite_times = [value for value in times if value is not None]
    duration = duration_s(rows)
    return {
        "input": str(path),
        "rows": len(rows),
        "duration_s": duration,
        "sample_rate_hz": (
            (len(rows) - 1) / duration
            if duration > 0 and len(rows) > 1
            else 0.0
        ),
        "invalid_time_samples": int(time_stats["nonfinite_samples"]),
        "time_monotonic": (
            len(finite_times) == len(rows)
            and all(
                current > previous
                for previous, current in zip(finite_times, finite_times[1:])
            )
        ),
        "max_position_span_m": max_span(
            stats,
            ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z"],
        ),
        "max_attitude_span_deg": max_span(
            stats,
            ["stabilizer.roll", "stabilizer.pitch"],
        ),
    }


def aggregate_signal_stats(
    logs: list[tuple[Path, list[dict[str, str]]]],
    column: str,
    *,
    max_range_span_mm: float,
) -> dict[str, float | int | None | list[str]]:
    segments = [(path, rows, signal_stats(rows, column)) for path, rows in logs]
    included = [
        (path, rows, segment)
        for path, rows, segment in segments
        if include_segment_for_noise(column, segment, max_range_span_mm)
    ]
    excluded = [
        (path, segment)
        for path, _rows, segment in segments
        if not include_segment_for_noise(column, segment, max_range_span_mm)
    ]
    all_values = [
        value
        for _path, rows, _segment in included
        for row in rows
        if (value := valid_column_float(row, column)) is not None
    ]
    stats = describe(all_values)
    valid_samples = sum(
        int(segment["samples"]) for _path, _rows, segment in included
    )
    raw_samples = sum(
        int(segment["raw_samples"]) for _path, _rows, segment in segments
    )
    invalid_samples = sum(
        int(segment["invalid_samples"]) for _path, _rows, segment in segments
    )
    nonfinite_samples = sum(
        int(segment.get("nonfinite_samples", 0))
        for _path, _rows, segment in segments
    )
    excluded_samples = sum(
        int(segment["samples"]) for _path, segment in excluded
    )
    if valid_samples:
        stats["std"] = pooled_std(
            [segment for _path, _rows, segment in included]
        )
        stats["global_span"] = stats["span"]
        stats["span"] = max(
            (
                float(segment["span"])
                for _path, _rows, segment in included
                if segment["span"] is not None
            ),
            default=0.0,
        )
    stats["raw_samples"] = raw_samples
    stats["invalid_samples"] = invalid_samples
    stats["nonfinite_samples"] = nonfinite_samples
    stats["excluded_samples"] = excluded_samples
    stats["excluded_segments"] = len(excluded)
    stats["excluded_inputs"] = [str(path) for path, _segment in excluded]
    stats["valid_ratio"] = valid_samples / raw_samples if raw_samples else None
    return stats


def include_segment_for_noise(
    column: str,
    segment: dict[str, float | int | None],
    max_range_span_mm: float,
) -> bool:
    if column not in RANGE_COLUMNS or int(segment["samples"]) <= 0:
        return True
    span = segment["span"]
    return span is None or float(span) <= max_range_span_mm


def signal_stats(
    rows: list[dict[str, str]],
    column: str,
) -> dict[str, float | int | None]:
    raw_entries = [row.get(column) for row in rows]
    raw_values = [
        value
        for item in raw_entries
        if item not in (None, "") and (value := _as_float(item)) is not None
    ]
    valid_values = [
        value for value in raw_values if valid_signal_value(column, value)
    ]
    stats = describe(valid_values)
    stats["raw_samples"] = len(raw_entries)
    stats["nonfinite_samples"] = sum(
        item not in (None, "") and _as_float(item) is None for item in raw_entries
    )
    stats["invalid_samples"] = len(raw_entries) - len(valid_values)
    stats["valid_ratio"] = (
        len(valid_values) / len(raw_entries) if raw_entries else None
    )
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
    variance = sum(
        int(segment["samples"]) * float(segment["std"] or 0.0) ** 2
        for segment in segments
    ) / total
    return sqrt(variance)


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "samples": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "span": None,
        }
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
    return max(
        [float(span) for span in spans if span is not None],
        default=0.0,
    )


def _as_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None
