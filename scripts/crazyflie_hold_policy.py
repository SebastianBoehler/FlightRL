from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import sleep, time

import torch

from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.hold_policy import (
    HOLD_LOG_VARIABLES,
    RangerHoldPolicy,
    command_from_hold_model,
    hold_command_row,
    hold_state_from_telemetry,
)
from flightrl.hardware.motion import (
    arm_crazyflie_for_flight,
    disarm_crazyflie_after_flight,
    install_legacy_hover_warning_filter,
)
from flightrl.hardware.preflight import require_supervisor_allows_flight
from flightrl.hardware.telemetry import build_log_configs, with_available_log_variables, with_extra_log_variables
from flightrl.sim2real.hardware_approval import HardwareApprovalError, require_hardware_approved


REQUIRED_POLICY_KEYS = frozenset(HOLD_LOG_VARIABLES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained Crazyflie ranger/attitude hold policy")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/ranger_hold.pt")
    parser.add_argument("--hardware-config", default="configs/hardware/crazyflie_2_1_brushless.toml")
    parser.add_argument("--output", default="artifacts/crazyflie_logs/ranger_hold_policy.csv")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--height-m", type=float, default=0.45)
    parser.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Explicit world target. Defaults to current x/y after takeoff.")
    parser.add_argument("--max-speed-m-s", type=float, default=0.25)
    parser.add_argument("--max-vertical-speed-m-s", type=float, default=0.18)
    parser.add_argument("--max-yawrate-deg-s", type=float, default=45.0)
    parser.add_argument("--confirm-flight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approval-manifest", default="artifacts/replay/sim2real_checkpoint_manifest_current_2026-05-20.json")
    args = parser.parse_args()

    if args.dry_run:
        model = load_policy(args.checkpoint)
        rows = [dry_run_row(model, args)]
    else:
        if not args.confirm_flight:
            raise SystemExit("--confirm-flight is required for real drone control")
        require_policy_approval(args.checkpoint, args.approval_manifest)
        model = load_policy(args.checkpoint)
        rows = run_live(model, args)
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


def load_policy(path: str | Path) -> RangerHoldPolicy:
    checkpoint = torch.load(path, map_location="cpu")
    hidden_size = int(checkpoint.get("hidden_size", 96))
    model = RangerHoldPolicy(hidden_size=hidden_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def require_policy_approval(checkpoint: str | Path, manifest: str | Path) -> None:
    try:
        record = require_hardware_approved(checkpoint, manifest)
    except HardwareApprovalError as exc:
        raise SystemExit(f"hardware approval blocked: {exc}") from exc
    print(f"hardware approval ok: task={record.get('task')} label={record.get('label')}", flush=True)


def dry_run_row(model: RangerHoldPolicy, args) -> dict[str, float]:
    target = target_tuple(args)
    telemetry = {
        "range.front": 180.0,
        "range.back": 2000.0,
        "range.left": 1500.0,
        "range.right": 1500.0,
        "range.up": 2000.0,
        "range.zrange": target[2] * 1000.0,
        "stateEstimate.z": target[2],
        "stabilizer.pitch": 35.0,
    }
    command = command_from_hold_model(model, hold_state_from_telemetry(telemetry, target)).clipped(
        max_speed=args.max_speed_m_s,
        max_vertical_speed=args.max_vertical_speed_m_s,
        max_yawrate=args.max_yawrate_deg_s,
    )
    return {"host_time_s": time(), **telemetry, **hold_command_row(command)}


def run_live(model: RangerHoldPolicy, args) -> list[dict[str, float]]:
    config = with_extra_log_variables(load_hardware_config(args.hardware_config), HOLD_LOG_VARIABLES)
    modules = require_cflib()
    install_legacy_hover_warning_filter()
    latest: dict[str, float] = {}
    rows: list[dict[str, float]] = []
    target = target_tuple(args) if args.target is not None else None
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
            if target is not None:
                print(f"hold policy loop started: duration_s={args.duration_s:.1f}, target={target}", flush=True)
            with modules.sync_logger_cls(scf, build_log_configs(modules, log_config)) as logger:
                while time() < deadline:
                    _timestamp, values, _conf = next(logger)
                    latest.update({key: float(value) for key, value in values.items()})
                    if not has_complete_policy_telemetry(latest):
                        continue
                    if target is None:
                        target = (_get(latest, "stateEstimate.x"), _get(latest, "stateEstimate.y"), args.height_m)
                        print(f"hold policy loop started: duration_s={args.duration_s:.1f}, target={target}", flush=True)
                    command = command_from_hold_model(model, hold_state_from_telemetry(latest, target)).clipped(
                        max_speed=args.max_speed_m_s,
                        max_vertical_speed=args.max_vertical_speed_m_s,
                        max_yawrate=args.max_yawrate_deg_s,
                    )
                    motion.start_linear_motion(command.vx_m_s, command.vy_m_s, command.vz_m_s, rate_yaw=command.yawrate_deg_s)
                    rows.append(
                        {
                            "host_time_s": time(),
                            "target_x_m": target[0],
                            "target_y_m": target[1],
                            "target_z_m": target[2],
                            **latest,
                            **hold_command_row(command),
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


def target_tuple(args) -> tuple[float, float, float]:
    values = args.target if args.target is not None else (0.0, 0.0, args.height_m)
    return (float(values[0]), float(values[1]), float(values[2]))


def has_complete_policy_telemetry(values: dict[str, float]) -> bool:
    return REQUIRED_POLICY_KEYS.issubset(values)


def _get(values: dict[str, float], key: str) -> float:
    return float(values.get(key, 0.0))


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
