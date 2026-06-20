from __future__ import annotations

import math
from queue import Empty

from flightrl.hardware.avoidance_policy import (
    AvoidanceCommand,
    RangerAvoidancePolicy,
    RangerReading,
    command_from_model,
    min_horizontal_range_m,
    min_horizontal_ttc_s,
    reactive_clearance_command,
    smooth_command,
)
from flightrl.hardware.avoidance_shadow import shadow_command_row
from flightrl.hardware.target_conditioned_policy import TargetSpec, command_from_target_model
from flightrl.hardware.target_direction import TargetDirectionConfig, target_direction_command
from flightrl.hardware.ttc_policy import TTCAvoidancePolicy, command_from_ttc_model


AVOIDANCE_LOG_VARIABLES = (
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "range.zrange",
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.vz",
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "gyro.x",
    "gyro.y",
    "gyro.z",
    "acc.x",
    "acc.y",
    "acc.z",
    "pm.vbat",
)


def build_control_command(model: RangerAvoidancePolicy | TTCAvoidancePolicy | None, reading, range_rate: RangerReading | None, args):
    if args.controller == "reactive":
        return reactive_clearance_command(
            reading,
            range_rate_m_s=range_rate,
            clearance_m=args.clearance_m,
            hard_clearance_m=args.hard_clearance_m,
            target_height_m=args.height_m,
            max_speed_m_s=args.max_speed_m_s,
            ttc_horizon_s=args.ttc_horizon_s,
            ttc_hard_s=args.ttc_hard_s,
            ttc_gain=args.ttc_gain,
        )
    if args.controller == "directional":
        return target_direction_command(
            reading,
            TargetDirectionConfig(
                direction_deg=args.target_direction_deg,
                target_speed_m_s=args.target_speed_m_s,
                clearance_m=args.clearance_m,
                hard_clearance_m=args.hard_clearance_m,
                target_height_m=args.height_m,
                avoidance_speed_m_s=args.max_speed_m_s,
                max_speed_m_s=args.max_speed_m_s,
                slowdown_gain=args.target_slowdown_gain,
                avoidance_gain=args.target_avoidance_gain,
                ttc_horizon_s=args.ttc_horizon_s,
                ttc_hard_s=args.ttc_hard_s,
                ttc_gain=args.ttc_gain,
            ),
            range_rate_m_s=range_rate,
        )
    if args.controller == "ttc-policy":
        if model is None:
            raise SystemExit("--checkpoint is required when --controller ttc-policy")
        return lock_height(command_from_ttc_model(model, reading, _rate_or_zero(range_rate), max_speed_m_s=args.max_speed_m_s), args)
    if model is None:
        raise SystemExit("--checkpoint is required when --controller policy")
    return command_from_model(model, reading, max_speed_m_s=args.max_speed_m_s)


def maybe_emergency_command(command: AvoidanceCommand, reading, range_rate: RangerReading | None, args) -> tuple[AvoidanceCommand, bool]:
    emergency = min_horizontal_range_m(reading) < args.emergency_clearance_m
    if args.emergency_ttc_s > 0.0 and min_horizontal_ttc_s(reading, range_rate) < args.emergency_ttc_s:
        emergency = True
    if not emergency:
        return lock_height(command, args), False
    return lock_height(
        reactive_clearance_command(
            reading,
            range_rate_m_s=range_rate,
            clearance_m=args.clearance_m,
            hard_clearance_m=args.hard_clearance_m,
            target_height_m=args.height_m,
            max_speed_m_s=args.emergency_max_speed_m_s,
            ttc_horizon_s=args.ttc_horizon_s,
            ttc_hard_s=args.ttc_hard_s,
            ttc_gain=args.ttc_gain,
        ),
        args,
    ), True


def build_shadow_command(model: RangerAvoidancePolicy | None, reading, args) -> AvoidanceCommand | None:
    if model is None:
        return None
    max_speed = args.shadow_max_speed_m_s if args.shadow_max_speed_m_s is not None else args.absolute_max_speed_m_s
    return lock_height(command_from_model(model, reading, max_speed_m_s=max_speed), args)


def build_target_shadow_command(model, reading, args) -> AvoidanceCommand | None:
    if model is None:
        return None
    target = TargetSpec(args.target_direction_deg, args.target_speed_m_s)
    return lock_height(command_from_target_model(model, reading, target, max_speed_m_s=args.target_shadow_max_speed_m_s), args)


def build_ttc_shadow_command(model, reading, range_rate: RangerReading | None, args) -> AvoidanceCommand | None:
    if model is None:
        return None
    return lock_height(
        command_from_ttc_model(model, reading, _rate_or_zero(range_rate), max_speed_m_s=args.ttc_shadow_max_speed_m_s),
        args,
    )


def lock_height(command: AvoidanceCommand, args) -> AvoidanceCommand:
    if not args.lock_height:
        return command
    return AvoidanceCommand(command.vx_m_s, command.vy_m_s, command.yawrate_deg_s, args.height_m)


def smooth_avoidance_command(command: AvoidanceCommand, previous: AvoidanceCommand, args, *, emergency: bool = False) -> AvoidanceCommand:
    return smooth_command(
        command,
        previous,
        alpha=args.smoothing_alpha,
        max_speed_step_m_s=args.emergency_speed_step_m_s if emergency else args.max_speed_step_m_s,
        max_yawrate_step_deg_s=args.max_yawrate_step_deg_s,
        max_zdistance_step_m=args.max_zdistance_step_m,
    )


def has_range_update(values: dict) -> bool:
    return any(key.startswith("range.") for key in values)


def safety_abort_reason(
    telemetry: dict[str, float],
    *,
    target_height_m: float,
    height_error_abort_m: float = 0.35,
    min_state_height_m: float = 0.10,
    max_state_height_m: float = 1.20,
) -> str | None:
    if _value(telemetry, "sys.isTumbled") > 0.5:
        return "tumbled"
    roll = abs(_value(telemetry, "stateEstimate.roll"))
    pitch = abs(_value(telemetry, "stateEstimate.pitch"))
    if roll > 45.0:
        return f"roll_gt_45deg:{roll:.1f}"
    if pitch > 35.0:
        return f"pitch_gt_35deg:{pitch:.1f}"
    gyro = math.sqrt(_value(telemetry, "gyro.x") ** 2 + _value(telemetry, "gyro.y") ** 2 + _value(telemetry, "gyro.z") ** 2)
    if gyro > 500.0:
        return f"gyro_gt_500dps:{gyro:.1f}"
    z = _value(telemetry, "stateEstimate.z", default=target_height_m)
    if z < min_state_height_m:
        return f"state_height_below_min:{z:.2f}"
    if z > max_state_height_m:
        return f"state_height_above_max:{z:.2f}"
    if height_error_abort_m > 0.0 and abs(z - target_height_m) > height_error_abort_m:
        return f"height_error_gt_{int(height_error_abort_m * 100)}cm:{z - target_height_m:.2f}"
    return None


def update_range_rate(
    reading: RangerReading,
    previous: RangerReading | None,
    previous_time_s: float | None,
    now_s: float,
    current_rate: RangerReading | None,
    *,
    alpha: float,
    max_abs_rate_m_s: float = 5.0,
) -> RangerReading | None:
    if previous is None or previous_time_s is None:
        return current_rate
    dt_s = max(now_s - previous_time_s, 1e-3)
    measured = RangerReading(
        front_m=_axis_rate(reading.front_m, previous.front_m, dt_s, max_abs_rate_m_s),
        back_m=_axis_rate(reading.back_m, previous.back_m, dt_s, max_abs_rate_m_s),
        left_m=_axis_rate(reading.left_m, previous.left_m, dt_s, max_abs_rate_m_s),
        right_m=_axis_rate(reading.right_m, previous.right_m, dt_s, max_abs_rate_m_s),
        up_m=_axis_rate(reading.up_m, previous.up_m, dt_s, max_abs_rate_m_s),
        zrange_m=_axis_rate(reading.zrange_m, previous.zrange_m, dt_s, max_abs_rate_m_s),
    )
    if current_rate is None:
        return measured
    a = max(0.0, min(1.0, alpha))
    return RangerReading(
        front_m=_blend_rate(current_rate.front_m, measured.front_m, a),
        back_m=_blend_rate(current_rate.back_m, measured.back_m, a),
        left_m=_blend_rate(current_rate.left_m, measured.left_m, a),
        right_m=_blend_rate(current_rate.right_m, measured.right_m, a),
        up_m=_blend_rate(current_rate.up_m, measured.up_m, a),
        zrange_m=_blend_rate(current_rate.zrange_m, measured.zrange_m, a),
    )


def raw_command_row(command: AvoidanceCommand) -> dict[str, float]:
    return {
        "raw_vx_m_s": command.vx_m_s,
        "raw_vy_m_s": command.vy_m_s,
        "raw_yawrate_deg_s": command.yawrate_deg_s,
        "raw_zdistance_m": command.zdistance_m,
    }


def range_rate_row(rate: RangerReading | None) -> dict[str, float]:
    if rate is None:
        return {}
    return {
        "range_rate_front_m_s": rate.front_m,
        "range_rate_back_m_s": rate.back_m,
        "range_rate_left_m_s": rate.left_m,
        "range_rate_right_m_s": rate.right_m,
        "range_rate_up_m_s": rate.up_m,
        "range_rate_zrange_m_s": rate.zrange_m,
    }


def next_log_sample(logger, *, timeout_s: float):
    queue = getattr(logger, "_queue", None)
    disconnect_event = getattr(logger, "DISCONNECT_EVENT", object())
    if queue is None:
        return next(logger)
    try:
        data = queue.get(timeout=timeout_s)
    except Empty:
        return None
    if data == disconnect_event:
        raise StopIteration
    return data


def _blend_rate(previous: float, measured: float, alpha: float) -> float:
    return previous + alpha * (measured - previous)


def _axis_rate(current_m: float, previous_m: float, dt_s: float, max_abs_rate_m_s: float) -> float:
    if current_m >= 3.95 or previous_m >= 3.95:
        return 0.0
    raw = (current_m - previous_m) / dt_s
    return max(-max_abs_rate_m_s, min(max_abs_rate_m_s, raw))


def _rate_or_zero(rate: RangerReading | None) -> RangerReading:
    return rate if rate is not None else RangerReading(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _value(values: dict[str, float], key: str, *, default: float = 0.0) -> float:
    try:
        return float(values.get(key, default))
    except (TypeError, ValueError):
        return default


__all__ = [
    "AVOIDANCE_LOG_VARIABLES",
    "build_control_command",
    "build_shadow_command",
    "build_target_shadow_command",
    "build_ttc_shadow_command",
    "has_range_update",
    "maybe_emergency_command",
    "next_log_sample",
    "range_rate_row",
    "raw_command_row",
    "safety_abort_reason",
    "shadow_command_row",
    "smooth_avoidance_command",
    "update_range_rate",
]
