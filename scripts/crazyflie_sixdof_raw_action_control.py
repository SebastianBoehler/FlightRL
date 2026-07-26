from __future__ import annotations

import argparse
import csv
from math import radians
from pathlib import Path
from time import sleep, time

import numpy as np
import torch

from flightrl.hardware.avoidance_live import AVOIDANCE_LOG_VARIABLES, next_log_sample
from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.motion import arm_crazyflie_for_flight, disarm_crazyflie_after_flight, install_legacy_hover_warning_filter
from flightrl.hardware.preflight import require_supervisor_allows_flight
from flightrl.hardware.sixdof_live_replay import live_env_from_telemetry, target_from_telemetry
from flightrl.hardware.sixdof_raw_action import RawPufferActionConfig, raw_action_to_manual_setpoint
from flightrl.hardware.telemetry import build_log_configs, with_available_log_variables, with_extra_log_variables
from flightrl.sim2real.hardware_approval import HardwareApprovalError, require_hardware_approved
from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.puffer_observation import scale_previous_action_observation
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy


LOG_VARIABLES = AVOIDANCE_LOG_VARIABLES + ("sys.isFlying", "sys.isTumbled", "pm.batteryLevel")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw Puffer six-DoF actions as Crazyflie manual rate setpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--approval-manifest", default="artifacts/replay/sim2real_checkpoint_manifest_current_2026-05-20.json")
    parser.add_argument("--hardware-config", default="configs/hardware/crazyflie_2_1_brushless.toml")
    parser.add_argument("--output", default="artifacts/crazyflie_logs/sixdof_puffer_raw_action_control.csv")
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--height-m", type=float, default=0.50)
    parser.add_argument("--startup-hover-s", type=float, default=1.5)
    parser.add_argument("--post-raw-hover-s", type=float, default=0.6)
    parser.add_argument("--hover-thrust-percent", type=float, default=49.0)
    parser.add_argument("--thrust-scale", type=float, default=0.75)
    parser.add_argument("--max-roll-rate-deg-s", type=float, default=343.7747)
    parser.add_argument("--max-pitch-rate-deg-s", type=float, default=343.7747)
    parser.add_argument("--max-yaw-rate-deg-s", type=float, default=229.1831)
    parser.add_argument("--target-yaw-deg", type=float, default=0.0)
    parser.add_argument("--target-mode", choices=("current_pose", "fixed_origin"), default="current_pose")
    parser.add_argument("--previous-action-observation-scale", type=float, default=0.25)
    parser.add_argument("--log-timeout-s", type=float, default=0.5)
    parser.add_argument("--replay-input")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-flight", action="store_true")
    parser.add_argument("--confirm-direct-action-control", action="store_true")
    parser.add_argument("--confirm-raw-puffer-output", action="store_true")
    args = parser.parse_args()

    policy = load_puffer_sixdof_policy(args.checkpoint)
    if policy.metadata.observation_dim != 28 or policy.metadata.action_dim != 4:
        raise SystemExit(f"unsupported Puffer shape: obs={policy.metadata.observation_dim} action={policy.metadata.action_dim}")
    raw_config = raw_config_from_args(args)
    if args.dry_run:
        rows = [control_row(policy, raw_config, synthetic_telemetry(args), args, np.zeros(4, dtype=np.float32), False, "dry_run", False)]
    elif args.replay_input:
        rows = replay_rows(policy, raw_config, args)
    else:
        if not args.confirm_flight or not args.confirm_direct_action_control or not args.confirm_raw_puffer_output:
            raise SystemExit("--confirm-flight, --confirm-direct-action-control and --confirm-raw-puffer-output are required")
        require_policy_approval(args.checkpoint, args.approval_manifest)
        rows = live_rows(policy, raw_config, args)
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"direct_action_control={not args.dry_run and not args.replay_input}")
    print("raw_puffer_output=True")


def raw_config_from_args(args) -> RawPufferActionConfig:
    return RawPufferActionConfig(
        hover_thrust_percent=args.hover_thrust_percent,
        thrust_scale=args.thrust_scale,
        max_roll_rate_deg_s=args.max_roll_rate_deg_s,
        max_pitch_rate_deg_s=args.max_pitch_rate_deg_s,
        max_yaw_rate_deg_s=args.max_yaw_rate_deg_s,
    )


def require_policy_approval(checkpoint: str | Path, manifest: str | Path) -> None:
    try:
        record = require_hardware_approved(checkpoint, manifest)
    except HardwareApprovalError as exc:
        raise SystemExit(f"hardware approval blocked: {exc}") from exc
    print(f"hardware approval ok: task={record.get('task')} label={record.get('label')}", flush=True)


def live_rows(policy, raw_config: RawPufferActionConfig, args) -> list[dict]:
    config = with_extra_log_variables(load_hardware_config(args.hardware_config), LOG_VARIABLES)
    modules = require_cflib()
    install_legacy_hover_warning_filter()
    latest: dict[str, float] = {}
    rows: list[dict] = []
    previous_action = np.zeros(4, dtype=np.float32)
    target_ref: list[np.ndarray | None] = [None]
    with sync_crazyflie_context(config, modules) as scf:
        log_config = with_available_log_variables(scf, config)
        commander = scf.cf.commander
        motion = modules.motion_commander_cls(scf, default_height=args.height_m)
        airborne = False
        try:
            require_supervisor_allows_flight(scf, modules, config)
            with modules.sync_logger_cls(scf, build_log_configs(modules, log_config)) as logger:
                arm_crazyflie_for_flight(scf.cf)
                motion.take_off(height=args.height_m, velocity=config.safety.velocity_m_s)
                airborne = True
                motion.stop()
                rows.extend(collect_rows(policy, raw_config, args, logger, latest, previous_action, target_ref, "startup_hover", args.startup_hover_s, controls_drone=True, send_raw=False))
                rows.extend(collect_rows(policy, raw_config, args, logger, latest, previous_action, target_ref, "raw_policy", args.duration_s, controls_drone=True, send_raw=True, commander=commander))
                stop_manual_setpoint(commander, raw_config.hover_thrust_percent)
                motion.stop()
                rows.extend(collect_rows(policy, raw_config, args, logger, latest, previous_action, target_ref, "post_raw_hover", args.post_raw_hover_s, controls_drone=True, send_raw=False))
        finally:
            if airborne:
                stop_manual_setpoint(commander, raw_config.hover_thrust_percent)
                motion.land(velocity=config.safety.velocity_m_s)
            commander.send_stop_setpoint()
            commander.send_notify_setpoint_stop()
            disarm_crazyflie_after_flight(scf.cf)
    return rows


def collect_rows(
    policy,
    raw_config: RawPufferActionConfig,
    args,
    logger,
    latest: dict[str, float],
    previous_action: np.ndarray,
    target_ref: list[np.ndarray | None],
    phase: str,
    duration_s: float,
    *,
    controls_drone: bool,
    send_raw: bool,
    commander=None,
) -> list[dict]:
    rows: list[dict] = []
    deadline = time() + max(0.0, duration_s)
    shadow_previous = previous_action.copy()
    while time() < deadline:
        sample = next_log_sample(logger, timeout_s=args.log_timeout_s)
        if sample is None:
            print(f"{phase} loop stopping: log timeout", flush=True)
            break
        _timestamp, values, _conf = sample
        latest.update({key: float(value) for key, value in values.items()})
        latest["host_time_s"] = time()
        if target_ref[0] is None:
            target_ref[0] = target_from_row(latest, args)
        source_previous = previous_action if send_raw else shadow_previous
        row = control_row(policy, raw_config, latest, args, source_previous, controls_drone, phase, send_raw, target=target_ref[0])
        action = np.asarray([row["action_thrust"], row["action_roll_rate"], row["action_pitch_rate"], row["action_yaw_rate"]], dtype=np.float32)
        if send_raw:
            if commander is None:
                raise RuntimeError("commander is required when send_raw=True")
            commander.send_setpoint_manual(
                row["roll_rate_deg_s"],
                row["commander_pitch_rate_deg_s"],
                row["yaw_rate_deg_s"],
                row["thrust_percent"],
                True,
            )
            previous_action[:] = action
        else:
            shadow_previous[:] = action
        rows.append(row)
    return rows


def replay_rows(policy, raw_config: RawPufferActionConfig, args) -> list[dict]:
    rows: list[dict] = []
    previous_action = np.zeros(4, dtype=np.float32)
    target = None
    latest: dict[str, float] = {}
    with Path(args.replay_input).open(newline="") as handle:
        for telemetry in csv.DictReader(handle):
            latest.update({key: parse_float(value) for key, value in telemetry.items() if value != ""})
            if target is None:
                target = target_from_row(latest, args)
            row = control_row(policy, raw_config, dict(latest), args, previous_action, False, "replay", False, target=target)
            previous_action[:] = [row["action_thrust"], row["action_roll_rate"], row["action_pitch_rate"], row["action_yaw_rate"]]
            rows.append(row)
    return rows


def control_row(policy, raw_config, telemetry: dict, args, previous_action: np.ndarray, controls_drone: bool, phase: str, raw_control_active: bool, *, target: np.ndarray | None = None) -> dict:
    env = SixDofCrazyflieEnv(num_envs=1, seed=0, task="obstacle_avoidance")
    fallback_target = np.asarray(target if target is not None else target_from_row(telemetry, args), dtype=np.float32)
    live_target = target_from_telemetry(telemetry, fallback_target)
    live_env_from_telemetry(env, telemetry, target=live_target, target_yaw=radians(args.target_yaw_deg))
    env.previous_action[0] = previous_action
    with torch.no_grad():
        observation = scale_previous_action_observation(env.observation(), getattr(args, "previous_action_observation_scale", 0.25))
        action = policy(torch.from_numpy(observation).float()).cpu().numpy()[0]
    setpoint = raw_action_to_manual_setpoint(action, raw_config)
    return {
        **telemetry,
        "host_time_s": float(telemetry.get("host_time_s", time()) or time()),
        "phase": phase,
        "controls_drone": controls_drone,
        "raw_control_active": raw_control_active,
        "raw_puffer_output": True,
        "target_x": float(live_target[0]),
        "target_y": float(live_target[1]),
        "target_z": float(live_target[2]),
        "action_thrust": float(action[0]),
        "action_roll_rate": float(action[1]),
        "action_pitch_rate": float(action[2]),
        "action_yaw_rate": float(action[3]),
        "roll_rate_deg_s": setpoint.roll_rate_deg_s,
        "pitch_rate_deg_s": setpoint.pitch_rate_deg_s,
        "commander_pitch_rate_deg_s": setpoint.commander_pitch_rate_deg_s,
        "yaw_rate_deg_s": setpoint.yaw_rate_deg_s,
        "thrust_percent": setpoint.thrust_percent,
    }


def stop_manual_setpoint(commander, hover_thrust_percent: float) -> None:
    commander.send_setpoint_manual(0.0, 0.0, 0.0, hover_thrust_percent, True)
    sleep(0.3)


def synthetic_telemetry(args) -> dict[str, float]:
    return {
        "range.front": 260.0,
        "range.back": 1800.0,
        "range.left": 900.0,
        "range.right": 900.0,
        "range.up": 1500.0,
        "range.zrange": args.height_m * 1000.0,
        "stateEstimate.z": args.height_m,
        "pm.vbat": 3.85,
        "host_time_s": time(),
    }


def target_from_row(row: dict[str, float], args) -> np.ndarray:
    if getattr(args, "target_mode", "current_pose") == "fixed_origin":
        return np.asarray([0.0, 0.0, args.height_m], dtype=np.float32)
    return np.asarray([float(row.get("stateEstimate.x", 0.0)), float(row.get("stateEstimate.y", 0.0)), args.height_m], dtype=np.float32)


def parse_float(raw) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def write_rows(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["host_time_s"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
