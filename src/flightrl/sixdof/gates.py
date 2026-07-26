from __future__ import annotations


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
    failures = []
    if metrics.get("clearance_p01_m", metrics["min_clearance_m"]) < min_clearance_m:
        failures.append("min_clearance")
    if metrics["mean_completed_fraction"] < min_completed_fraction:
        failures.append("completion")
    if metrics["mean_position_error_m"] > max_position_error_m:
        failures.append("position_error")
    if max_yaw_error_rad is not None and metrics.get("mean_yaw_error_rad", 0.0) > max_yaw_error_rad:
        failures.append("yaw_error")
    if max_yaw_p95_error_rad is not None and metrics.get("yaw_error_p95_rad", 0.0) > max_yaw_p95_error_rad:
        failures.append("yaw_error_p95")
    if max_settled_yaw_p95_error_rad is not None and metrics.get("settled_yaw_error_p95_rad", 0.0) > max_settled_yaw_p95_error_rad:
        failures.append("settled_yaw_error_p95")
    if max_horizontal_speed_p95_m_s is not None and metrics.get("horizontal_speed_p95_m_s", 0.0) > max_horizontal_speed_p95_m_s:
        failures.append("horizontal_speed_p95")
    if (
        max_open_space_horizontal_speed_p95_m_s is not None
        and metrics.get("open_space_horizontal_speed_p95_m_s", 0.0) > max_open_space_horizontal_speed_p95_m_s
    ):
        failures.append("open_space_horizontal_speed_p95")
    if max_tilt_p95_deg is not None and metrics.get("tilt_p95_deg", 0.0) > max_tilt_p95_deg:
        failures.append("tilt_p95")
    return {"passed": not failures, "failures": failures}
