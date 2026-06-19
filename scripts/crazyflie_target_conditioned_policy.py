from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import sleep, time

from flightrl.hardware.avoidance_policy import (
    AvoidanceCommand,
    command_row,
    min_horizontal_range_m,
    reactive_clearance_command,
    reading_from_telemetry,
    smooth_command,
    vertical_velocity_from_height_error,
)
from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.motion import arm_crazyflie_for_flight, disarm_crazyflie_after_flight, install_legacy_hover_warning_filter
from flightrl.hardware.preflight import require_supervisor_allows_flight
from flightrl.hardware.target_conditioned_policy import TargetSpec, command_from_target_model, load_target_policy
from flightrl.hardware.telemetry import build_log_configs, with_available_log_variables, with_extra_log_variables
from flightrl.sim2real.hardware_approval import HardwareApprovalError, require_hardware_approved


LOG_VARIABLES = (
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
    parser = argparse.ArgumentParser(description="Run a target-conditioned checkpoint on Crazyflie hover setpoints")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--approval-manifest", default="artifacts/replay/sim2real_checkpoint_manifest_current_2026-05-20.json")
    parser.add_argument("--hardware-config", default="configs/hardware/crazyflie_2_1_brushless.toml")
    parser.add_argument("--output", default="artifacts/crazyflie_logs/target_conditioned_policy.csv")
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--height-m", type=float, default=0.50)
    parser.add_argument("--target-direction-deg", type=float, default=45.0)
    parser.add_argument("--target-speed-m-s", type=float, default=0.16)
    parser.add_argument("--max-speed-m-s", type=float, default=0.30)
    parser.add_argument("--max-speed-step-m-s", type=float, default=0.08)
    parser.add_argument("--max-vertical-speed-m-s", type=float, default=0.18)
    parser.add_argument("--safety-clearance-m", type=float, default=0.55)
    parser.add_argument("--safety-speed-m-s", type=float, default=0.45)
    parser.add_argument("--projected-stop-m", type=float, default=0.55)
    parser.add_argument("--projected-clearance-m", type=float, default=1.10)
    parser.add_argument("--abort-clearance-m", type=float, default=0.10)
    parser.add_argument("--confirm-flight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model = load_target_policy(args.checkpoint)
    target = TargetSpec(args.target_direction_deg, args.target_speed_m_s)
    if args.dry_run:
        reading = reading_from_telemetry({"range.front": 1200.0, "range.back": 1600.0, "range.left": 1400.0, "range.right": 1500.0, "range.zrange": args.height_m * 1000.0})
        command, safety_override, projection_scale = build_command(model, reading, target, args, AvoidanceCommand(0.0, 0.0, 0.0, args.height_m))
        print(f"dry_run target checkpoint command: {command} safety_override={safety_override} projection_scale={projection_scale:.3f}")
        return
    if not args.confirm_flight:
        raise SystemExit("--confirm-flight is required for real drone control")
    require_policy_approval(args.checkpoint, args.approval_manifest)
    rows = run_live(model, target, args)
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


def require_policy_approval(checkpoint: str | Path, manifest: str | Path) -> None:
    try:
        record = require_hardware_approved(checkpoint, manifest)
    except HardwareApprovalError as exc:
        raise SystemExit(f"hardware approval blocked: {exc}") from exc
    print(f"hardware approval ok: task={record.get('task')} label={record.get('label')}", flush=True)


def run_live(model, target: TargetSpec, args) -> list[dict[str, float]]:
    config = with_extra_log_variables(load_hardware_config(args.hardware_config), LOG_VARIABLES)
    modules = require_cflib()
    install_legacy_hover_warning_filter()
    latest: dict[str, float] = {}
    rows: list[dict[str, float]] = []
    previous = AvoidanceCommand(0.0, 0.0, 0.0, args.height_m)
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
            deadline = time() + args.duration_s
            with modules.sync_logger_cls(scf, build_log_configs(modules, log_config)) as logger:
                while time() < deadline:
                    _timestamp, values, _conf = next(logger)
                    latest.update({key: float(value) for key, value in values.items()})
                    reading = reading_from_telemetry(latest)
                    if should_abort_clearance(reading, args):
                        rows.append(abort_row(latest))
                        break
                    command, safety_override, projection_scale = build_command(model, reading, target, args, previous)
                    previous = command
                    vz_m_s = vertical_velocity_from_height_error(command, reading, max_vertical_speed_m_s=args.max_vertical_speed_m_s)
                    motion.start_linear_motion(command.vx_m_s, command.vy_m_s, vz_m_s, rate_yaw=command.yawrate_deg_s)
                    rows.append(
                        {
                            "host_time_s": time(),
                            **latest,
                            **command_row(command),
                            "projection_scale": projection_scale,
                            "safety_override": float(safety_override),
                            "vz_m_s": vz_m_s,
                        }
                    )
        finally:
            if airborne:
                motion.stop()
                sleep(0.5)
                motion.land(velocity=config.safety.velocity_m_s)
            commander.send_stop_setpoint()
            commander.send_notify_setpoint_stop()
            disarm_crazyflie_after_flight(scf.cf)
    return rows


def should_abort_clearance(reading, args) -> bool:
    return min_horizontal_range_m(reading) < args.abort_clearance_m


def abort_row(latest: dict[str, float]) -> dict[str, float]:
    return {
        "host_time_s": time(),
        **latest,
        **command_row(AvoidanceCommand(0.0, 0.0, 0.0, float(latest.get("range.zrange", 500.0)) / 1000.0)),
        "abort_clearance": 1.0,
        "projection_scale": 0.0,
        "safety_override": 1.0,
        "vz_m_s": 0.0,
    }


def build_command(model, reading, target: TargetSpec, args, previous: AvoidanceCommand) -> tuple[AvoidanceCommand, bool, float]:
    safety_override = min_horizontal_range_m(reading) < args.safety_clearance_m
    if safety_override:
        raw = reactive_clearance_command(
            reading,
            clearance_m=max(args.safety_clearance_m, args.safety_clearance_m + 0.05),
            hard_clearance_m=0.10,
            target_height_m=args.height_m,
            max_speed_m_s=args.safety_speed_m_s,
        )
        projection_scale = 1.0
    else:
        raw = command_from_target_model(model, reading, target, max_speed_m_s=args.max_speed_m_s)
        raw, projection_scale = apply_projected_clearance(raw, reading, args)
    locked = AvoidanceCommand(raw.vx_m_s, raw.vy_m_s, raw.yawrate_deg_s, args.height_m)
    command = smooth_command(
        locked,
        previous,
        alpha=1.0,
        max_speed_step_m_s=args.safety_speed_m_s if safety_override else args.max_speed_step_m_s,
        max_yawrate_step_deg_s=6.0,
        max_zdistance_step_m=0.02,
    ).clipped(max_speed=args.safety_speed_m_s if safety_override else args.max_speed_m_s, max_yawrate=45.0)
    return command, safety_override, projection_scale


def apply_projected_clearance(command: AvoidanceCommand, reading, args) -> tuple[AvoidanceCommand, float]:
    vx_scale = projected_axis_scale(command.vx_m_s, positive_range_m=reading.front_m, negative_range_m=reading.back_m, args=args)
    vy_scale = projected_axis_scale(command.vy_m_s, positive_range_m=reading.left_m, negative_range_m=reading.right_m, args=args)
    return (
        AvoidanceCommand(
            vx_m_s=command.vx_m_s * vx_scale,
            vy_m_s=command.vy_m_s * vy_scale,
            yawrate_deg_s=command.yawrate_deg_s,
            zdistance_m=command.zdistance_m,
        ),
        min(vx_scale, vy_scale),
    )


def projected_axis_scale(value: float, *, positive_range_m: float, negative_range_m: float, args) -> float:
    if value > 0.0:
        return clearance_scale(positive_range_m, stop_m=args.projected_stop_m, clear_m=args.projected_clearance_m)
    if value < 0.0:
        return clearance_scale(negative_range_m, stop_m=args.projected_stop_m, clear_m=args.projected_clearance_m)
    return 1.0


def clearance_scale(distance_m: float, *, stop_m: float, clear_m: float) -> float:
    if clear_m <= stop_m:
        raise ValueError("projected_clearance_m must be greater than projected_stop_m")
    return max(0.0, min(1.0, (distance_m - stop_m) / (clear_m - stop_m)))


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
