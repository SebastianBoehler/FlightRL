from __future__ import annotations

from flightrl.evidence_values import finite_number


def gate_status(
    metrics: dict,
    *,
    min_clearance_m: float,
    min_completed_fraction: float,
    max_position_error_m: float,
    max_yaw_error_rad: float | None = None,
    max_yaw_p95_error_rad: float | None = None,
    max_settled_yaw_p95_error_rad: float | None = None,
    max_horizontal_speed_p95_m_s: float | None = None,
    max_open_space_horizontal_speed_p95_m_s: float | None = None,
    max_tilt_p95_deg: float | None = None,
) -> dict:
    failures: list[str] = []
    if not valid_thresholds(
        min_clearance_m=min_clearance_m,
        min_completed_fraction=min_completed_fraction,
        max_position_error_m=max_position_error_m,
        optional=(
            max_yaw_error_rad,
            max_yaw_p95_error_rad,
            max_settled_yaw_p95_error_rad,
            max_horizontal_speed_p95_m_s,
            max_open_space_horizontal_speed_p95_m_s,
            max_tilt_p95_deg,
        ),
    ):
        return {"passed": False, "failures": ["thresholds_invalid"]}
    clearance = metric_value(metrics, "clearance_p01_m", "min_clearance_m")
    check_min(clearance, min_clearance_m, "min_clearance", failures)
    check_min(
        metric_value(metrics, "mean_completed_fraction"),
        min_completed_fraction,
        "completion",
        failures,
    )
    check_max(
        metric_value(metrics, "mean_position_error_m"),
        max_position_error_m,
        "position_error",
        failures,
    )
    optional_maxima = (
        ("mean_yaw_error_rad", max_yaw_error_rad, "yaw_error"),
        ("yaw_error_p95_rad", max_yaw_p95_error_rad, "yaw_error_p95"),
        (
            "settled_yaw_error_p95_rad",
            max_settled_yaw_p95_error_rad,
            "settled_yaw_error_p95",
        ),
        (
            "horizontal_speed_p95_m_s",
            max_horizontal_speed_p95_m_s,
            "horizontal_speed_p95",
        ),
        (
            "open_space_horizontal_speed_p95_m_s",
            max_open_space_horizontal_speed_p95_m_s,
            "open_space_horizontal_speed_p95",
        ),
        ("tilt_p95_deg", max_tilt_p95_deg, "tilt_p95"),
    )
    for metric, maximum, label in optional_maxima:
        if maximum is not None:
            check_max(metric_value(metrics, metric), maximum, label, failures)
    return {"passed": not failures, "failures": failures}


def metric_value(metrics: dict, key: str, fallback: str | None = None) -> float | None:
    if key in metrics:
        return finite_number(metrics[key])
    if fallback is not None and fallback in metrics:
        return finite_number(metrics[fallback])
    return None


def check_min(
    value: float | None,
    threshold: float,
    label: str,
    failures: list[str],
) -> None:
    if value is None:
        failures.append(f"{label}_invalid")
    elif value < threshold:
        failures.append(label)


def check_max(
    value: float | None,
    threshold: float,
    label: str,
    failures: list[str],
) -> None:
    if value is None:
        failures.append(f"{label}_invalid")
    elif value > threshold:
        failures.append(label)


def valid_thresholds(
    *,
    min_clearance_m: object,
    min_completed_fraction: object,
    max_position_error_m: object,
    optional: tuple[object | None, ...],
) -> bool:
    clearance = finite_number(min_clearance_m)
    completion = finite_number(min_completed_fraction)
    position = finite_number(max_position_error_m)
    if (
        clearance is None
        or clearance < 0.0
        or completion is None
        or not 0.0 <= completion <= 1.0
        or position is None
        or position < 0.0
    ):
        return False
    return all(
        value is None
        or (
            (parsed := finite_number(value)) is not None
            and parsed >= 0.0
        )
        for value in optional
    )
