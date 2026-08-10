from __future__ import annotations

from math import isfinite
from typing import Mapping, Sequence

from .flight_telemetry import (
    HORIZONTAL_CLEARANCE_VARIABLES,
    MINIMUM_CLEARANCE_M,
    RANGER_NO_RETURN_MINIMUM_MM,
)
from .ranger_integrity import ranger_row_integrity
from .ranger_map import summarize_map
from .ranger_projection import (
    points_from_rows,
    prepare_rows,
    rows_with_mapping_time,
    trajectory_from_rows,
)
from .ranger_schema import RANGER_KEYS


MINIMUM_FLOW_QUALITY = 80
MINIMUM_FLOW_QUALITY_RATIO = 0.80


def validate_range_flight(rows: Sequence[Mapping[str, float]]) -> dict[str, object]:
    integrity = ranger_row_integrity(rows)
    device_timed_rows, _time_source = rows_with_mapping_time(rows)
    prepared = prepare_rows(device_timed_rows, min_drone_z_m=0.20)
    points = points_from_rows(prepared)
    trajectory = trajectory_from_rows(prepared)
    mapping = summarize_map(
        points,
        trajectory,
        min_points=100,
        min_duration_s=10.0,
        min_horizontal_sensors=3,
        min_trajectory_xy_span_m=0.25,
        min_yaw_span_deg=15.0,
        max_step_speed_m_s=2.0,
        source_integrity=integrity,
    )
    range_values_valid = bool(rows) and all(
        _uint_value(row.get(variable), maximum=65535)
        for row in rows
        for variable in RANGER_KEYS
    )
    clearance_ok = bool(prepared) and all(
        _clearance_value_ok(row.get(variable))
        for row in prepared
        for variable in HORIZONTAL_CLEARANCE_VARIABLES
    )
    flow_values_valid = bool(prepared) and all(
        _uint_value(row.get("motion.motion"), maximum=255)
        and _uint_value(row.get("motion.squal"), maximum=255)
        for row in prepared
    )
    quality_ratio = (
        sum(float(row["motion.squal"]) >= MINIMUM_FLOW_QUALITY for row in prepared)
        / len(prepared)
        if prepared and flow_values_valid
        else 0.0
    )
    motion_status_ratio = (
        sum(int(float(row["motion.motion"])) == 0xB0 for row in prepared)
        / len(prepared)
        if prepared and flow_values_valid
        else 0.0
    )
    checks = {
        "source_integrity": integrity["valid"] is True,
        "range_values": range_values_valid,
        "clearance": clearance_ok,
        "flow_values": flow_values_valid,
        "flow_quality": quality_ratio >= MINIMUM_FLOW_QUALITY_RATIO,
        "mapping": mapping["mapping_ready"] is True,
    }
    return {
        "range_calibration_passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "mapping": mapping,
        "flow": {
            "minimum_quality": MINIMUM_FLOW_QUALITY,
            "quality_ratio": quality_ratio,
            "motion_status_b0_ratio": motion_status_ratio,
        },
    }


def _uint_value(value: object, *, maximum: int) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(numeric) and numeric.is_integer() and 0.0 <= numeric <= maximum


def _clearance_value_ok(value: object) -> bool:
    if not _uint_value(value, maximum=65535):
        return False
    distance_mm = float(value)
    return (
        distance_mm >= RANGER_NO_RETURN_MINIMUM_MM
        or distance_mm >= MINIMUM_CLEARANCE_M * 1000.0
    )
