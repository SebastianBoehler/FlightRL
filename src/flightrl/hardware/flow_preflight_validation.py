from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


SCHEMA = "flightrl.aideck_flow_preflight_validation.v1"
HEALTHY_MOTION_STATUS = 0xB0
MINIMUM_ROWS = 80
MINIMUM_DURATION_S = 4.0
MINIMUM_HEALTHY_MOTION_ROWS = 5
MINIMUM_HEALTHY_MOTION_RATIO = 0.80
MINIMUM_HEALTHY_STATUS_RATIO = 0.80
MINIMUM_STATUS_QUALITY_RATIO = 0.80
MINIMUM_SQUAL = 80.0
REQUIRED_COLUMNS = (
    "host_time_s",
    "crazyflie_time_ms",
    "motion.motion",
    "motion.deltaX",
    "motion.deltaY",
    "motion.squal",
    "range.zrange",
)


def validate_flow_preflight(path: str | Path) -> dict[str, object]:
    source = Path(path)
    values = _load_columns(source)
    host_times = values["host_time_s"]
    device_times = values["crazyflie_time_ms"]
    statuses = values["motion.motion"]
    delta_x = values["motion.deltaX"]
    delta_y = values["motion.deltaY"]
    squal = values["motion.squal"]
    zrange = values["range.zrange"]

    host_gaps = np.diff(host_times)
    device_gaps = np.diff(device_times)
    moving = (delta_x != 0.0) | (delta_y != 0.0)
    status_valid = bool(
        np.all(statuses == np.floor(statuses))
        and np.all((0.0 <= statuses) & (statuses <= 255.0))
    )
    healthy_motion = moving & (statuses == HEALTHY_MOTION_STATUS)
    healthy_status = statuses == HEALTHY_MOTION_STATUS
    status_quality = healthy_status & (squal >= MINIMUM_SQUAL)
    moving_rows = int(moving.sum())
    healthy_rows = int(healthy_motion.sum())
    healthy_ratio = healthy_rows / max(1, moving_rows)
    healthy_status_rows = int(healthy_status.sum())
    healthy_status_ratio = healthy_status_rows / len(statuses)
    status_quality_rows = int(status_quality.sum())
    status_quality_ratio = status_quality_rows / len(statuses)
    minimum_healthy_squal = (
        float(squal[healthy_motion].min()) if healthy_rows else 0.0
    )
    duration_s = float(host_times[-1] - host_times[0])

    checks = {
        "telemetry_rows": len(host_times) >= MINIMUM_ROWS,
        "capture_duration": duration_s >= MINIMUM_DURATION_S,
        "host_time_order": bool(np.all(host_gaps > 0.0)),
        "device_time_order": bool(np.all(device_gaps > 0.0)),
        "telemetry_gap": bool(np.all(host_gaps <= 0.075))
        and bool(np.all(device_gaps <= 75.0)),
        "status_values": status_valid,
        "healthy_status": status_valid
        and healthy_status_ratio >= MINIMUM_HEALTHY_STATUS_RATIO,
        "healthy_motion": status_valid
        and healthy_rows >= MINIMUM_HEALTHY_MOTION_ROWS
        and healthy_ratio >= MINIMUM_HEALTHY_MOTION_RATIO,
        "flow_quality": healthy_rows >= MINIMUM_HEALTHY_MOTION_ROWS
        and minimum_healthy_squal >= MINIMUM_SQUAL
        and status_quality_ratio >= MINIMUM_STATUS_QUALITY_RATIO,
        "zrange_plausible": bool(np.all((zrange > 20.0) & (zrange < 4000.0))),
    }
    metrics = {
        "rows": len(host_times),
        "duration_s": duration_s,
        "maximum_host_gap_s": float(host_gaps.max()),
        "maximum_device_gap_ms": float(device_gaps.max()),
        "moving_rows": moving_rows,
        "healthy_motion_rows": healthy_rows,
        "healthy_motion_ratio": healthy_ratio,
        "healthy_status_rows": healthy_status_rows,
        "healthy_status_ratio": healthy_status_ratio,
        "healthy_status_quality_rows": status_quality_rows,
        "healthy_status_quality_ratio": status_quality_ratio,
        "minimum_healthy_motion_squal": minimum_healthy_squal,
        "minimum_zrange_mm": float(zrange.min()),
        "maximum_zrange_mm": float(zrange.max()),
    }
    return {
        "schema": SCHEMA,
        "source": str(source),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "metrics": metrics,
        "flow_preflight_passed": all(checks.values()),
        "controls_drone": False,
        "flight_authority": False,
        "authority_reason": (
            "This verifies a short props-off raw Flow/Z-ranger response only; it does "
            "not establish estimator, camera, policy, shadow, deployment, or flight readiness."
        ),
    }


def _load_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2 or not all(name in rows[0] for name in REQUIRED_COLUMNS):
        raise ValueError("Flow preflight CSV is missing rows or required columns")
    try:
        values = {
            name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
            for name in REQUIRED_COLUMNS
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("Flow preflight CSV contains a nonnumeric value") from exc
    if not all(np.isfinite(column).all() for column in values.values()):
        raise ValueError("Flow preflight CSV contains nonfinite values")
    return values
