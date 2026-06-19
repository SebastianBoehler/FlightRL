from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import sleep, time

from flightrl.hardware.avoidance_policy import (
    AvoidanceCommand,
    RangerAvoidancePolicy,
    command_from_model,
    command_row,
    min_horizontal_range_m,
    reactive_clearance_command,
    reading_from_telemetry,
    smooth_command,
    vertical_velocity_from_height_error,
)
from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.avoidance_shadow import load_ranger_policy, shadow_command_row
from flightrl.hardware.motion import (
    arm_crazyflie_for_flight,
    disarm_crazyflie_after_flight,
    install_legacy_hover_warning_filter,
)
from flightrl.hardware.preflight import require_supervisor_allows_flight
from flightrl.hardware.avoidance_state import EscapeHoldState
from flightrl.hardware.target_direction import TargetDirectionConfig, target_direction_command
from flightrl.hardware.target_conditioned_policy import TargetSpec, command_from_target_model, load_target_policy
from flightrl.hardware.telemetry import build_log_configs, with_available_log_variables, with_extra_log_variables
from flightrl.sim2real.hardware_approval import HardwareApprovalError, require_hardware_approved


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained ranger avoidance policy on Crazyflie hover setpoints")
    parser.add_argument("--checkpoint")
    parser.add_argument("--shadow-checkpoint")
    parser.add_argument("--shadow-max-speed-m-s", type=float, default=None)
    parser.add_argument("--target-shadow-checkpoint")
    parser.add_argument("--target-shadow-max-speed-m-s", type=float, default=0.90)
    parser.add_argument("--controller", choices=("policy", "reactive", "directional"), default="policy")
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
    parser.add_argument("--max-vertical-speed-m-s", type=float, default=0.18)
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--max-speed-step-m-s", type=float, default=0.03)
    parser.add_argument("--max-yawrate-step-deg-s", type=float, default=6.0)
    parser.add_argument("--max-zdistance-step-m", type=float, default=0.04)
    parser.add_argument("--emergency-clearance-m", type=float, default=0.25)
    parser.add_argument("--emergency-max-speed-m-s", type=float, default=0.14)
    parser.add_argument("--emergency-speed-step-m-s", type=float, default=0.06)
    parser.add_argument("--absolute-max-speed-m-s", type=float, default=0.35)
    parser.add_argument("--emergency-hold-steps", type=int, default=0)
    parser.add_argument("--lock-height", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confirm-flight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approval-manifest", default="artifacts/replay/sim2real_checkpoint_manifest_current_2026-05-20.json")
    args = parser.parse_args()

    if args.controller == "policy" and not args.checkpoint:
        raise SystemExit("--checkpoint is required when --controller policy")
    if args.dry_run:
        model = load_policy(args.checkpoint) if args.controller == "policy" else None
        shadow_model = load_policy(args.shadow_checkpoint) if args.shadow_checkpoint else None
        target_shadow_model = load_target_policy(args.target_shadow_checkpoint) if args.target_shadow_checkpoint else None
        reading = reading_from_telemetry({"range.front": 250.0, "range.back": 2000.0, "range.zrange": args.height_m * 1000.0})
        command = build_command(model, reading, args)
        command, emergency = maybe_emergency_command(command, reading, args)
        smoothed = smooth_avoidance_command(command, AvoidanceCommand(0.0, 0.0, 0.0, args.height_m), args, emergency=emergency)
        shadow = build_shadow_command(shadow_model, reading, args) if shadow_model else None
        target_shadow = build_target_shadow_command(target_shadow_model, reading, args) if target_shadow_model else None
        print(f"dry_run avoidance command: raw={command} smoothed={smoothed} shadow={shadow} target_shadow={target_shadow}")
        return
    if not args.confirm_flight:
        raise SystemExit("--confirm-flight is required for real drone control")
    if args.controller == "policy":
        require_policy_approval(args.checkpoint, args.approval_manifest)
    model = load_policy(args.checkpoint) if args.controller == "policy" else None
    shadow_model = load_policy(args.shadow_checkpoint) if args.shadow_checkpoint else None
    target_shadow_model = load_target_policy(args.target_shadow_checkpoint) if args.target_shadow_checkpoint else None
    rows = run_live(model, shadow_model, target_shadow_model, args)
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


def load_policy(path: str | Path) -> RangerAvoidancePolicy:
    return load_ranger_policy(path)


def require_policy_approval(checkpoint: str | Path, manifest: str | Path) -> None:
    try:
        record = require_hardware_approved(checkpoint, manifest)
    except HardwareApprovalError as exc:
        raise SystemExit(f"hardware approval blocked: {exc}") from exc
    print(f"hardware approval ok: task={record.get('task')} label={record.get('label')}", flush=True)


def run_live(model: RangerAvoidancePolicy | None, shadow_model: RangerAvoidancePolicy | None, target_shadow_model, args) -> list[dict[str, float]]:
    config = with_extra_log_variables(load_hardware_config(args.hardware_config), AVOIDANCE_LOG_VARIABLES)
    modules = require_cflib()
    install_legacy_hover_warning_filter()
    latest: dict[str, float] = {}
    rows: list[dict[str, float]] = []
    previous_command = AvoidanceCommand(0.0, 0.0, 0.0, args.height_m)
    escape_hold = EscapeHoldState(hold_steps=args.emergency_hold_steps)
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
                    _timestamp, values, _conf = next(logger)
                    latest.update({key: float(value) for key, value in values.items()})
                    reading = reading_from_telemetry(latest)
                    command = build_command(model, reading, args)
                    command, emergency = maybe_emergency_command(command, reading, args)
                    shadow = build_shadow_command(shadow_model, reading, args) if shadow_model else None
                    target_shadow = build_target_shadow_command(target_shadow_model, reading, args) if target_shadow_model else None
                    speed_limit = args.emergency_max_speed_m_s if emergency else args.max_speed_m_s
                    command = command.clipped(
                        max_speed=min(speed_limit, args.absolute_max_speed_m_s),
                        max_yawrate=config.safety.turn_rate_deg_s,
                    )
                    command, escape_hold_active = escape_hold.update(command, emergency=emergency)
                    smoothed_command = smooth_avoidance_command(command, previous_command, args, emergency=emergency).clipped(
                        max_speed=min(speed_limit, args.absolute_max_speed_m_s),
                        max_yawrate=config.safety.turn_rate_deg_s,
                    )
                    previous_command = smoothed_command
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
                            "host_time_s": time(),
                            **latest,
                            **command_row(smoothed_command),
                            **raw_command_row(command),
                            "emergency_active": float(emergency),
                            "escape_hold_active": float(escape_hold_active),
                            "min_horizontal_range_m": min_horizontal_range_m(reading),
                            "vz_m_s": vz_m_s,
                    }
                    if shadow is not None:
                        row.update(shadow_command_row(shadow))
                        row["shadow_monitor_only"] = 1.0
                    if target_shadow is not None:
                        row.update(shadow_command_row(target_shadow, prefix="target_shadow"))
                        row["target_shadow_monitor_only"] = 1.0
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


def build_command(model: RangerAvoidancePolicy | None, reading, args):
    if args.controller == "reactive":
        return reactive_clearance_command(
            reading,
            clearance_m=args.clearance_m,
            hard_clearance_m=args.hard_clearance_m,
            target_height_m=args.height_m,
            max_speed_m_s=args.max_speed_m_s,
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
            ),
        )
    if model is None:
        raise SystemExit("--checkpoint is required when --controller policy")
    return command_from_model(model, reading, max_speed_m_s=args.max_speed_m_s)


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


def maybe_emergency_command(command: AvoidanceCommand, reading, args) -> tuple[AvoidanceCommand, bool]:
    emergency = min_horizontal_range_m(reading) < args.emergency_clearance_m
    if not emergency:
        return lock_height(command, args), False
    return lock_height(reactive_clearance_command(
        reading,
        clearance_m=args.clearance_m,
        hard_clearance_m=args.hard_clearance_m,
        target_height_m=args.height_m,
        max_speed_m_s=args.emergency_max_speed_m_s,
    ), args), True


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


def raw_command_row(command: AvoidanceCommand) -> dict[str, float]:
    return {
        "raw_vx_m_s": command.vx_m_s,
        "raw_vy_m_s": command.vy_m_s,
        "raw_yawrate_deg_s": command.yawrate_deg_s,
        "raw_zdistance_m": command.zdistance_m,
    }


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
