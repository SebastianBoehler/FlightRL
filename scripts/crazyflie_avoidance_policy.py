from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import sleep, time

from flightrl.hardware.avoidance_policy import (
    AvoidanceCommand,
    RangerReading,
    RangerAvoidancePolicy,
    clip_horizontal_norm,
    command_row,
    min_horizontal_range_m,
    min_horizontal_ttc_s,
    reading_from_telemetry,
    vertical_velocity_from_clearance,
    vertical_velocity_from_height_error,
)
from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.avoidance_close_escape import apply_close_escape_correction
from flightrl.hardware.avoidance_live import (
    AVOIDANCE_LOG_VARIABLES,
    build_control_command,
    build_shadow_command,
    build_target_shadow_command,
    build_ttc_shadow_command,
    has_range_update,
    maybe_emergency_command,
    next_log_sample,
    range_rate_row,
    raw_command_row,
    safety_abort_reason,
    shadow_command_row,
    smooth_avoidance_command,
    update_range_rate,
)
from flightrl.hardware.avoidance_shadow import load_ranger_policy
from flightrl.hardware.motion import (
    arm_crazyflie_for_flight,
    disarm_crazyflie_after_flight,
    install_legacy_hover_warning_filter,
)
from flightrl.hardware.preflight import require_supervisor_allows_flight
from flightrl.hardware.avoidance_state import DirectionHoldState, EscapeHoldState
from flightrl.hardware.target_conditioned_policy import load_target_policy
from flightrl.hardware.telemetry import build_log_configs, with_available_log_variables, with_extra_log_variables
from flightrl.hardware.ttc_policy import load_ttc_policy
from flightrl.sim2real.hardware_approval import HardwareApprovalError, require_hardware_approved


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained ranger avoidance policy on Crazyflie hover setpoints")
    parser.add_argument("--checkpoint")
    parser.add_argument("--shadow-checkpoint")
    parser.add_argument("--shadow-max-speed-m-s", type=float, default=None)
    parser.add_argument("--target-shadow-checkpoint")
    parser.add_argument("--target-shadow-max-speed-m-s", type=float, default=0.90)
    parser.add_argument("--ttc-shadow-checkpoint")
    parser.add_argument("--ttc-shadow-max-speed-m-s", type=float, default=0.65)
    parser.add_argument("--controller", choices=("policy", "ttc-policy", "reactive", "directional"), default="policy")
    parser.add_argument("--hardware-config", default="configs/hardware/crazyflie_2_1_brushless.toml")
    parser.add_argument("--output", default="artifacts/crazyflie_logs/avoidance_policy.csv")
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--height-m", type=float, default=0.45)
    parser.add_argument("--clearance-m", type=float, default=0.45)
    parser.add_argument("--hard-clearance-m", type=float, default=0.10)
    parser.add_argument("--max-speed-m-s", type=float, default=0.25)
    parser.add_argument("--target-direction-deg", type=float, default=0.0)
    parser.add_argument("--target-speed-m-s", type=float, default=0.18)
    parser.add_argument("--target-slowdown-gain", type=float, default=0.85)
    parser.add_argument("--target-avoidance-gain", type=float, default=1.0)
    parser.add_argument("--ttc-horizon-s", type=float, default=0.0)
    parser.add_argument("--ttc-hard-s", type=float, default=0.12)
    parser.add_argument("--ttc-gain", type=float, default=1.0)
    parser.add_argument("--range-rate-alpha", type=float, default=0.65)
    parser.add_argument("--range-rate-max-m-s", type=float, default=5.0)
    parser.add_argument("--log-timeout-s", type=float, default=0.5)
    parser.add_argument("--max-vertical-speed-m-s", type=float, default=0.18)
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--max-speed-step-m-s", type=float, default=0.03)
    parser.add_argument("--max-yawrate-step-deg-s", type=float, default=6.0)
    parser.add_argument("--max-zdistance-step-m", type=float, default=0.04)
    parser.add_argument("--emergency-clearance-m", type=float, default=0.25)
    parser.add_argument("--emergency-ttc-s", type=float, default=0.0)
    parser.add_argument("--emergency-max-speed-m-s", type=float, default=0.14)
    parser.add_argument("--emergency-speed-step-m-s", type=float, default=0.06)
    parser.add_argument("--absolute-max-speed-m-s", type=float, default=0.35)
    parser.add_argument("--emergency-hold-steps", type=int, default=0)
    parser.add_argument("--anti-oscillation-hold-s", type=float, default=0.0)
    parser.add_argument("--anti-oscillation-min-speed-m-s", type=float, default=0.12)
    parser.add_argument("--anti-oscillation-hard-clearance-m", type=float, default=0.11)
    parser.add_argument("--anti-oscillation-hard-ttc-s", type=float, default=0.25)
    parser.add_argument("--close-escape-clearance-m", type=float, default=0.0)
    parser.add_argument("--close-escape-min-speed-m-s", type=float, default=0.0)
    parser.add_argument("--close-escape-brake-gain", type=float, default=0.0)
    parser.add_argument("--close-escape-brake-max-m-s", type=float, default=0.0)
    parser.add_argument("--height-error-abort-m", type=float, default=0.35, help="Target-relative height abort. Set 0 to disable for vertical-clearance experiments.")
    parser.add_argument("--min-state-height-m", type=float, default=0.10)
    parser.add_argument("--max-state-height-m", type=float, default=1.20)
    parser.add_argument("--vertical-controller", choices=("height", "clearance"), default="height")
    parser.add_argument("--lock-height", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confirm-flight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approval-manifest", default="artifacts/replay/sim2real_checkpoint_manifest_current_2026-05-20.json")
    args = parser.parse_args()

    if args.controller in ("policy", "ttc-policy") and not args.checkpoint:
        raise SystemExit(f"--checkpoint is required when --controller {args.controller}")
    if args.dry_run:
        model = load_policy(args.checkpoint, args.controller) if args.controller in ("policy", "ttc-policy") else None
        shadow_model = load_policy(args.shadow_checkpoint) if args.shadow_checkpoint else None
        target_shadow_model = load_target_policy(args.target_shadow_checkpoint) if args.target_shadow_checkpoint else None
        ttc_shadow_model = load_ttc_policy(args.ttc_shadow_checkpoint) if args.ttc_shadow_checkpoint else None
        reading = reading_from_telemetry({"range.front": 250.0, "range.back": 2000.0, "range.zrange": args.height_m * 1000.0})
        range_rate = RangerReading(front_m=-0.6, back_m=0.0, left_m=0.0, right_m=0.0, up_m=0.0, zrange_m=0.0)
        command = build_control_command(model, reading, range_rate, args)
        command, emergency = maybe_emergency_command(command, reading, range_rate, args)
        smoothed = smooth_avoidance_command(command, AvoidanceCommand(0.0, 0.0, 0.0, args.height_m), args, emergency=emergency)
        shadow = build_shadow_command(shadow_model, reading, args) if shadow_model else None
        target_shadow = build_target_shadow_command(target_shadow_model, reading, args) if target_shadow_model else None
        ttc_shadow = build_ttc_shadow_command(ttc_shadow_model, reading, range_rate, args) if ttc_shadow_model else None
        print(f"dry_run avoidance command: raw={command} smoothed={smoothed} shadow={shadow} target_shadow={target_shadow} ttc_shadow={ttc_shadow}")
        return
    if not args.confirm_flight:
        raise SystemExit("--confirm-flight is required for real drone control")
    if args.controller in ("policy", "ttc-policy"):
        require_policy_approval(args.checkpoint, args.approval_manifest)
    model = load_policy(args.checkpoint, args.controller) if args.controller in ("policy", "ttc-policy") else None
    shadow_model = load_policy(args.shadow_checkpoint) if args.shadow_checkpoint else None
    target_shadow_model = load_target_policy(args.target_shadow_checkpoint) if args.target_shadow_checkpoint else None
    ttc_shadow_model = load_ttc_policy(args.ttc_shadow_checkpoint) if args.ttc_shadow_checkpoint else None
    rows = run_live(model, shadow_model, target_shadow_model, ttc_shadow_model, args)
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


def load_policy(path: str | Path, controller: str = "policy"):
    return load_ttc_policy(path) if controller == "ttc-policy" else load_ranger_policy(path)


def require_policy_approval(checkpoint: str | Path, manifest: str | Path) -> None:
    try:
        record = require_hardware_approved(checkpoint, manifest)
    except HardwareApprovalError as exc:
        raise SystemExit(f"hardware approval blocked: {exc}") from exc
    print(f"hardware approval ok: task={record.get('task')} label={record.get('label')}", flush=True)


def run_live(model, shadow_model: RangerAvoidancePolicy | None, target_shadow_model, ttc_shadow_model, args) -> list[dict[str, float]]:
    config = with_extra_log_variables(load_hardware_config(args.hardware_config), AVOIDANCE_LOG_VARIABLES)
    modules = require_cflib()
    install_legacy_hover_warning_filter()
    latest: dict[str, float] = {}
    rows: list[dict[str, float]] = []
    previous_command = AvoidanceCommand(0.0, 0.0, 0.0, args.height_m)
    previous_reading: RangerReading | None = None
    previous_reading_time_s: float | None = None
    range_rate: RangerReading | None = None
    escape_hold = EscapeHoldState(hold_steps=args.emergency_hold_steps)
    direction_hold = DirectionHoldState(
        hold_s=args.anti_oscillation_hold_s,
        min_speed_m_s=args.anti_oscillation_min_speed_m_s,
        hard_clearance_m=args.anti_oscillation_hard_clearance_m,
        hard_ttc_s=args.anti_oscillation_hard_ttc_s,
    )
    with sync_crazyflie_context(config, modules) as scf:
        log_config = with_available_log_variables(scf, config)
        commander = scf.cf.commander
        motion = modules.motion_commander_cls(scf, default_height=args.height_m)
        airborne = False
        try:
            require_supervisor_allows_flight(scf, modules, config)
            arm_crazyflie_for_flight(scf.cf)
            motion.take_off(height=args.height_m, velocity=config.safety.velocity_m_s)
            airborne = True
            motion.stop()
            print(f"policy loop started: duration_s={args.duration_s:.1f}, height_m={args.height_m:.2f}", flush=True)
            deadline = time() + args.duration_s
            with modules.sync_logger_cls(scf, build_log_configs(modules, log_config)) as logger:
                while time() < deadline:
                    sample = next_log_sample(logger, timeout_s=args.log_timeout_s)
                    if sample is None:
                        print("policy loop stopping: log timeout", flush=True)
                        break
                    _timestamp, values, _conf = sample
                    now_s = time()
                    latest.update({key: float(value) for key, value in values.items()})
                    reading = reading_from_telemetry(latest)
                    abort_reason = safety_abort_reason(
                        latest,
                        target_height_m=args.height_m,
                        height_error_abort_m=args.height_error_abort_m,
                        min_state_height_m=args.min_state_height_m,
                        max_state_height_m=args.max_state_height_m,
                    )
                    if abort_reason is not None:
                        print(f"policy loop stopping: safety abort {abort_reason}", flush=True)
                        if abort_reason == "tumbled":
                            airborne = False
                        break
                    if has_range_update(values):
                        range_rate = update_range_rate(
                            reading,
                            previous_reading,
                            previous_reading_time_s,
                            now_s,
                            range_rate,
                            alpha=args.range_rate_alpha,
                            max_abs_rate_m_s=args.range_rate_max_m_s,
                        )
                        previous_reading = reading
                        previous_reading_time_s = now_s
                    command = build_control_command(model, reading, range_rate, args)
                    command, emergency = maybe_emergency_command(command, reading, range_rate, args)
                    shadow = build_shadow_command(shadow_model, reading, args) if shadow_model else None
                    target_shadow = build_target_shadow_command(target_shadow_model, reading, args) if target_shadow_model else None
                    ttc_shadow = build_ttc_shadow_command(ttc_shadow_model, reading, range_rate, args) if ttc_shadow_model else None
                    speed_limit = args.emergency_max_speed_m_s if emergency else args.max_speed_m_s
                    command = command.clipped(
                        max_speed=min(speed_limit, args.absolute_max_speed_m_s),
                        max_yawrate=config.safety.turn_rate_deg_s,
                    )
                    command, escape_hold_active = escape_hold.update(command, emergency=emergency)
                    raw_command = command
                    command, direction_hold_active = direction_hold.update(
                        command,
                        now_s=now_s,
                        reading=reading,
                        range_rate=range_rate,
                    )
                    command, close_escape = apply_close_escape_correction(
                        command,
                        reading,
                        latest,
                        clearance_m=args.close_escape_clearance_m,
                        min_speed_m_s=args.close_escape_min_speed_m_s,
                        brake_gain=args.close_escape_brake_gain,
                        brake_max_m_s=args.close_escape_brake_max_m_s,
                    )
                    command = clip_horizontal_norm(
                        command,
                        max_speed=min(speed_limit, args.absolute_max_speed_m_s),
                        max_yawrate=config.safety.turn_rate_deg_s,
                    )
                    smoothed_command = smooth_avoidance_command(command, previous_command, args, emergency=emergency)
                    smoothed_command = clip_horizontal_norm(
                        smoothed_command,
                        max_speed=min(speed_limit, args.absolute_max_speed_m_s),
                        max_yawrate=config.safety.turn_rate_deg_s,
                    )
                    previous_command = smoothed_command
                    if args.vertical_controller == "clearance":
                        vz_m_s = vertical_velocity_from_clearance(
                            reading,
                            top_clearance_m=args.clearance_m,
                            bottom_clearance_m=0.35,
                            hard_clearance_m=args.hard_clearance_m,
                            max_vertical_speed_m_s=args.max_vertical_speed_m_s,
                        )
                    else:
                        vz_m_s = vertical_velocity_from_height_error(
                            smoothed_command,
                            reading,
                            max_vertical_speed_m_s=args.max_vertical_speed_m_s,
                        )
                    motion.start_linear_motion(
                        smoothed_command.vx_m_s,
                        smoothed_command.vy_m_s,
                        vz_m_s,
                        rate_yaw=smoothed_command.yawrate_deg_s,
                    )
                    row = {
                            "host_time_s": now_s,
                            **latest,
                            **command_row(smoothed_command),
                            **raw_command_row(raw_command),
                            **range_rate_row(range_rate),
                            "emergency_active": float(emergency),
                            "escape_hold_active": float(escape_hold_active),
                            "direction_hold_active": float(direction_hold_active),
                            "min_horizontal_range_m": min_horizontal_range_m(reading),
                            "min_horizontal_ttc_s": min_horizontal_ttc_s(reading, range_rate),
                            "vz_m_s": vz_m_s,
                            **close_escape.row(),
                    }
                    if shadow is not None:
                        row.update(shadow_command_row(shadow))
                        row["shadow_monitor_only"] = 1.0
                    if target_shadow is not None:
                        row.update(shadow_command_row(target_shadow, prefix="target_shadow"))
                        row["target_shadow_monitor_only"] = 1.0
                    if ttc_shadow is not None:
                        row.update(shadow_command_row(ttc_shadow, prefix="ttc_shadow"))
                        row["ttc_shadow_monitor_only"] = 1.0
                    rows.append(row)
        finally:
            if airborne:
                motion.stop()
                sleep(0.5)
                motion.land(velocity=config.safety.velocity_m_s)
            commander.send_stop_setpoint()
            commander.send_notify_setpoint_stop()
            disarm_crazyflie_after_flight(scf.cf)
    return rows


def write_rows(path: str | Path, rows: list[dict[str, float]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["host_time_s"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
