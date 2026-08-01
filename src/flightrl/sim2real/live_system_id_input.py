from __future__ import annotations

import csv
import math
from pathlib import Path


REQUIRED_TRACKING_COLUMNS = (
    "host_time_s",
    "vx_m_s",
    "vy_m_s",
    "stateEstimate.vx",
    "stateEstimate.vy",
)
YAW_COLUMNS = ("stabilizer.yaw", "stateEstimate.yaw")


def read_tracking_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = [key for key in REQUIRED_TRACKING_COLUMNS if key not in columns]
        yaw_columns = tuple(key for key in YAW_COLUMNS if key in columns)
        if missing or not yaw_columns:
            required = [*missing, *(YAW_COLUMNS if not yaw_columns else ())]
            raise ValueError(
                f"{path}: missing required tracking columns: {', '.join(required)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: tracking log has no rows")
    for row_index, row in enumerate(rows, start=2):
        for key in REQUIRED_TRACKING_COLUMNS:
            if finite_float(row.get(key)) is None:
                raise ValueError(
                    f"{path}:{row_index}: invalid required tracking value for {key}"
                )
        if not any(finite_float(row.get(key)) is not None for key in yaw_columns):
            raise ValueError(
                f"{path}:{row_index}: invalid required tracking value for yaw"
            )
    times = [float(row["host_time_s"]) for row in rows]
    if any(
        current <= previous
        for previous, current in zip(times, times[1:], strict=False)
    ):
        raise ValueError(f"{path}: host_time_s must be strictly increasing")
    return rows


def finite_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
