from __future__ import annotations

from math import isfinite
from typing import Mapping, Sequence

from .ranger_schema import RANGER_KEYS, RANGER_POSE_KEYS


def ranger_row_integrity(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    required = (*RANGER_POSE_KEYS, *RANGER_KEYS)
    columns = set(rows[0]) if rows else set()
    missing_columns = [key for key in required if key not in columns]
    invalid_pose_rows = 0
    invalid_ranger_values = 0
    for row in rows:
        if any(_finite_value(row.get(key)) is None for key in RANGER_POSE_KEYS):
            invalid_pose_rows += 1
        invalid_ranger_values += sum(
            _finite_value(row.get(key)) is None
            for key in RANGER_KEYS
        )
    failures = []
    if missing_columns:
        failures.append("missing_source_columns")
    if invalid_pose_rows:
        failures.append("invalid_source_pose")
    if invalid_ranger_values:
        failures.append("invalid_source_ranger")
    return {
        "present": True,
        "valid": not failures,
        "rows": len(rows),
        "missing_columns": missing_columns,
        "invalid_pose_rows": invalid_pose_rows,
        "invalid_ranger_values": invalid_ranger_values,
        "failures": failures,
    }


def _finite_value(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None
