from __future__ import annotations

from queue import Empty
from math import hypot
from time import sleep, time

import numpy as np

from .avoidance_live import next_log_sample, safety_abort_reason
from .errors import HardwareSafetyError
from .motion import disarm_crazyflie_after_flight
from .preflight import inspect_decks
from .visual_waypoint import (
    StraightWaypoint,
    goal_intent,
    waypoint_envelope_abort_reason,
)


VISUAL_WAYPOINT_LOG_VARIABLES = (
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.vz",
    "stateEstimate.roll",
    "stateEstimate.pitch",
    "stateEstimate.yaw",
    "gyro.x",
    "gyro.y",
    "gyro.z",
    "range.zrange",
    "motion.motion",
    "motion.deltaX",
    "motion.deltaY",
    "motion.squal",
    "motion.shutter",
    "motion.outlierCount",
    "motion.std",
    "ctrltarget.vx",
    "ctrltarget.vy",
    "controller.cmd_roll",
    "controller.cmd_pitch",
    "controller.cmd_thrust",
    "motor.m1",
    "motor.m2",
    "motor.m3",
    "motor.m4",
    "pm.vbat",
    "pm.batteryLevel",
    "sys.isFlying",
    "sys.isTumbled",
)


def visual_live_abort_reason(
    telemetry,
    prediction,
    waypoint,
    waypoint_config,
    live_config,
) -> str | None:
    reason = safety_abort_reason(
        telemetry,
        target_height_m=waypoint_config.height_m,
        height_error_abort_m=0.20,
        min_state_height_m=0.15,
        max_state_height_m=0.85,
    )
    if reason is None:
        reason = waypoint_envelope_abort_reason(
            position_from_telemetry(telemetry),
            waypoint,
            waypoint_config,
        )
    if reason is not None:
        return reason
    return visual_input_abort_reason(prediction, live_config)


def visual_hover_abort_reason(
    telemetry,
    prediction,
    origin_xy_m,
    waypoint_config,
    live_config,
) -> str | None:
    reason = safety_abort_reason(
        telemetry,
        target_height_m=waypoint_config.height_m,
        height_error_abort_m=0.20,
        min_state_height_m=0.15,
        max_state_height_m=0.85,
    )
    if reason is not None:
        return reason
    displacement = hypot(
        float(telemetry["stateEstimate.x"]) - origin_xy_m[0],
        float(telemetry["stateEstimate.y"]) - origin_xy_m[1],
    )
    if displacement > live_config.max_hover_displacement_m:
        return (
            f"firmware_hover_displacement_gt_"
            f"{live_config.max_hover_displacement_m:.2f}m:{displacement:.3f}"
        )
    speed = hypot(
        float(telemetry.get("stateEstimate.vx", 0.0)),
        float(telemetry.get("stateEstimate.vy", 0.0)),
    )
    if speed > live_config.max_hover_speed_m_s:
        return (
            f"firmware_hover_speed_gt_"
            f"{live_config.max_hover_speed_m_s:.2f}m_s:{speed:.3f}"
        )
    return visual_input_abort_reason(prediction, live_config)


def visual_input_abort_reason(prediction, live_config) -> str | None:
    if time() - float(prediction["worker_host_time_s"]) > live_config.max_camera_age_s:
        return "camera_stale"
    if float(prediction["frame_mean"]) < live_config.min_frame_mean:
        return "camera_too_dark"
    if float(prediction["input_contrast_std"]) < live_config.min_input_contrast:
        return "camera_low_contrast"
    if int(prediction["dropped_frames"]) > live_config.max_dropped_frames:
        return "camera_drop_budget_exceeded"
    actions = [
        float(prediction[f"action_{axis}"])
        for axis in ("vx", "vy", "vz", "yaw")
    ]
    return None if np.isfinite(actions).all() else "non_finite_policy_action"


def require_camera_gate(camera, live_config) -> None:
    if float(camera["frame_mean"]) < live_config.min_frame_mean:
        raise HardwareSafetyError("AI Deck frame is too dark for active control")
    if float(camera["input_contrast_std"]) < live_config.min_input_contrast:
        raise HardwareSafetyError("AI Deck frame contrast is below the active gate")


def require_flow_deck(scf, config) -> None:
    report = inspect_decks(scf, config)
    if not report.ok:
        details = "; ".join((*report.warnings, *report.details.values()))
        raise HardwareSafetyError(f"Flow Deck preflight failed: {details}")


def wait_for_telemetry(logger, latest, timeout_s: float) -> None:
    deadline = time() + 3.0
    required = (
        "stateEstimate.x",
        "stateEstimate.y",
        "stateEstimate.z",
        "stateEstimate.yaw",
        "pm.vbat",
    )
    while time() < deadline:
        update_telemetry(logger, latest, timeout_s)
        if all(name in latest for name in required):
            return
    raise RuntimeError("live telemetry did not become usable before arming")


def wait_for_takeoff(logger, latest, height_m: float, timeout_s: float) -> None:
    drain_telemetry(logger, latest)
    deadline = time() + 2.0
    streak = 0
    while time() < deadline:
        update_telemetry(logger, latest, timeout_s)
        height_error = abs(latest.get("stateEstimate.z", 0.0) - height_m)
        airborne = (
            latest.get("sys.isFlying", 0.0) > 0.5
            and height_error <= 0.12
            and abs(latest.get("stateEstimate.roll", 0.0)) <= 15.0
            and abs(latest.get("stateEstimate.pitch", 0.0)) <= 15.0
        )
        streak = streak + 1 if airborne else 0
        if streak >= 3:
            return
    raise HardwareSafetyError("takeoff telemetry gate did not pass")


def drain_telemetry(logger, latest) -> int:
    queue = getattr(logger, "_queue", None)
    if queue is None:
        return 0
    disconnect_event = getattr(logger, "DISCONNECT_EVENT", object())
    drained = 0
    while True:
        try:
            sample = queue.get_nowait()
        except Empty:
            break
        if sample == disconnect_event:
            raise HardwareSafetyError("Crazyflie disconnected during takeoff")
        timestamp, values, conf = sample
        merge_telemetry_sample(latest, timestamp, values, conf)
        drained += 1
    return drained


def update_telemetry(logger, latest, timeout_s: float) -> None:
    sample = next_log_sample(logger, timeout_s=timeout_s)
    if sample is None:
        raise HardwareSafetyError("telemetry timeout")
    timestamp, values, conf = sample
    merge_telemetry_sample(latest, timestamp, values, conf)


def merge_telemetry_sample(latest, timestamp, values, conf) -> None:
    latest.update({key: float(value) for key, value in values.items()})
    latest["crazyflie_time_ms"] = int(timestamp)
    if conf is not None:
        latest["telemetry_log_block"] = getattr(conf, "name", str(conf))


def position_from_telemetry(values) -> tuple[float, float, float]:
    return (
        float(values["stateEstimate.x"]),
        float(values["stateEstimate.y"]),
        float(values["stateEstimate.z"]),
    )


def intent_from_telemetry(values, waypoint: StraightWaypoint) -> np.ndarray:
    return goal_intent(
        position_from_telemetry(values),
        float(values["stateEstimate.yaw"]),
        waypoint,
    )


def visual_control_row(
    phase,
    telemetry,
    prediction,
    *,
    command=None,
    controls_drone,
) -> dict:
    return {
        "host_time_s": time(),
        "phase": phase,
        **telemetry,
        **prediction,
        **(command or {}),
        "controls_drone": controls_drone,
        "monitor_only": not controls_drone,
    }


def shutdown_visual_flight(
    cf,
    commander,
    motion,
    *,
    airborne: bool,
    landing_velocity_m_s: float,
) -> None:
    landing_error: Exception | None = None
    try:
        if airborne:
            motion.start_linear_motion(0.0, 0.0, 0.0, rate_yaw=0.0)
            sleep(0.8)
            motion.stop()
            motion.land(velocity=landing_velocity_m_s)
    except Exception as exc:
        landing_error = exc
    finally:
        try:
            commander.send_stop_setpoint()
        finally:
            try:
                commander.send_notify_setpoint_stop()
            finally:
                disarm_crazyflie_after_flight(cf)
    if landing_error is not None:
        raise HardwareSafetyError(
            f"landing sequence failed: {landing_error}"
        ) from landing_error
