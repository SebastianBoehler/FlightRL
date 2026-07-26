from __future__ import annotations

import argparse
import csv
from math import radians
from pathlib import Path
from time import sleep, time

import numpy as np
import torch

from flightrl.hardware.avoidance_live import AVOIDANCE_LOG_VARIABLES, next_log_sample, safety_abort_reason
from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.motion import arm_crazyflie_for_flight, disarm_crazyflie_after_flight, install_legacy_hover_warning_filter
from flightrl.hardware.preflight import require_supervisor_allows_flight
from flightrl.hardware.sixdof_velocity_adapter import SixDofVelocityAdapterConfig, sixdof_action_to_velocity_command
from flightrl.hardware.sixdof_velocity_shield import (
    SixDofVelocityShieldConfig,
    SixDofVelocityShieldState,
    apply_sixdof_velocity_shield,
    sixdof_shield_row,
)
from flightrl.hardware.telemetry import build_log_configs, with_available_log_variables, with_extra_log_variables
from flightrl.sim2real.hardware_approval import HardwareApprovalError, require_hardware_approved
from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.env import euler_to_quat
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy


LOG_VARIABLES = AVOIDANCE_LOG_VARIABLES + ("sys.isFlying", "sys.isTumbled", "pm.batteryLevel")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Puffer six-DoF checkpoint through the Crazyflie velocity adapter.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--approval-manifest", default="artifacts/replay/sim2real_checkpoint_manifest_current_2026-05-20.json")
    parser.add_argument("--hardware-config", default="configs/hardware/crazyflie_2_1_brushless.toml")
    parser.add_argument("--output", default="artifacts/crazyflie_logs/sixdof_puffer_velocity_control.csv")
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--height-m", type=float, default=0.50)
    parser.add_argument("--max-horizontal-speed-m-s", type=float, default=0.12)
    parser.add_argument("--max-vertical-speed-m-s", type=float, default=0.04)
    parser.add_argument("--max-yawrate-deg-s", type=float, default=12.0)
    parser.add_argument("--policy-blend", type=float, default=1.0)
    parser.add_argument("--rate-horizon-s", type=float, default=0.08)
    parser.add_argument("--max-virtual-tilt-rad", type=float, default=0.18)
    parser.add_argument("--horizontal-gain-s", type=float, default=0.06)
    parser.add_argument("--shield-clearance-m", type=float, default=0.45)
    parser.add_argument("--shield-hard-clearance-m", type=float, default=0.10)
    parser.add_argument("--shield-max-speed-m-s", type=float, default=0.35)
    parser.add_argument("--shield-ttc-horizon-s", type=float, default=0.80)
    parser.add_argument("--shield-ttc-hard-s", type=float, default=0.15)
    parser.add_argument("--shield-ttc-gain", type=float, default=1.2)
    parser.add_argument("--range-rate-alpha", type=float, default=0.65)
    parser.add_argument("--range-rate-max-m-s", type=float, default=5.0)
    parser.add_argument("--log-timeout-s", type=float, default=0.5)
    parser.add_argument("--height-error-abort-m", type=float, default=0.0)
    parser.add_argument("--min-state-height-m", type=float, default=-10.0)
    parser.add_argument("--max-state-height-m", type=float, default=10.0)
    parser.add_argument("--raw-policy-control", action="store_true", help="Disable deterministic shield and software abort thresholds.")
    parser.add_argument("--replay-input")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-flight", action="store_true")
    parser.add_argument("--confirm-direct-policy-control", action="store_true")
    parser.add_argument("--confirm-raw-policy-control", action="store_true")
    args = parser.parse_args()

    policy = load_puffer_sixdof_policy(args.checkpoint)
    if policy.metadata.observation_dim != 28 or policy.metadata.action_dim != 4:
        raise SystemExit(f"unsupported Puffer six-DoF shape: obs={policy.metadata.observation_dim} action={policy.metadata.action_dim}")
    adapter = adapter_config(args)
    shield = shield_config(args)
    if args.dry_run:
        rows = [control_row(policy, adapter, shield, SixDofVelocityShieldState(), synthetic_telemetry(args), args, None, time(), None, controls_drone=False)]
    elif args.replay_input:
        rows = replay_rows(policy, adapter, shield, args)
    else:
        if not args.confirm_flight or not args.confirm_direct_policy_control:
            raise SystemExit("--confirm-flight and --confirm-direct-policy-control are required for live policy control")
        if args.raw_policy_control and not args.confirm_raw_policy_control:
            raise SystemExit("--confirm-raw-policy-control is required when --raw-policy-control disables deterministic guards")
        require_policy_approval(args.checkpoint, args.approval_manifest)
        rows = live_rows(policy, adapter, shield, args)
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"direct_policy_control={not args.dry_run and not args.replay_input}")
    print(f"raw_policy_control={args.raw_policy_control}")


def adapter_config(args) -> SixDofVelocityAdapterConfig:
    return SixDofVelocityAdapterConfig(
        max_horizontal_speed_m_s=args.max_horizontal_speed_m_s,
        max_vertical_speed_m_s=args.max_vertical_speed_m_s,
        max_yawrate_deg_s=args.max_yawrate_deg_s,
        rate_horizon_s=args.rate_horizon_s,
        max_virtual_tilt_rad=args.max_virtual_tilt_rad,
        horizontal_gain_s=args.horizontal_gain_s,
        policy_blend=args.policy_blend,
    )


def shield_config(args) -> SixDofVelocityShieldConfig:
    return SixDofVelocityShieldConfig(
        clearance_m=args.shield_clearance_m,
        hard_clearance_m=args.shield_hard_clearance_m,
        max_speed_m_s=args.shield_max_speed_m_s,
        ttc_horizon_s=args.shield_ttc_horizon_s,
        ttc_hard_s=args.shield_ttc_hard_s,
        ttc_gain=args.shield_ttc_gain,
        range_rate_alpha=args.range_rate_alpha,
        range_rate_max_m_s=args.range_rate_max_m_s,
    )


def require_policy_approval(checkpoint: str | Path, manifest: str | Path) -> None:
    try:
        record = require_hardware_approved(checkpoint, manifest)
    except HardwareApprovalError as exc:
        raise SystemExit(f"hardware approval blocked: {exc}") from exc
    print(f"hardware approval ok: task={record.get('task')} label={record.get('label')}", flush=True)


def live_rows(policy, adapter: SixDofVelocityAdapterConfig, shield: SixDofVelocityShieldConfig, args) -> list[dict]:
    config = with_extra_log_variables(load_hardware_config(args.hardware_config), LOG_VARIABLES)
    modules = require_cflib()
    install_legacy_hover_warning_filter()
    latest: dict[str, float] = {}
    rows: list[dict] = []
    anchor_xy: tuple[float, float] | None = None
    previous_action = np.zeros(4, dtype=np.float32)
    shield_state = SixDofVelocityShieldState()
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
                    sample = next_log_sample(logger, timeout_s=args.log_timeout_s)
                    if sample is None:
                        print("sixdof velocity loop stopping: log timeout", flush=True)
                        break
                    _timestamp, values, _conf = sample
                    now_s = time()
                    latest.update({key: float(value) for key, value in values.items()})
                    latest["host_time_s"] = now_s
                    if not args.raw_policy_control:
                        reason = safety_abort_reason(
                            latest,
                            target_height_m=args.height_m,
                            height_error_abort_m=args.height_error_abort_m,
                            min_state_height_m=args.min_state_height_m,
                            max_state_height_m=args.max_state_height_m,
                        )
                        if reason is not None:
                            print(f"sixdof velocity loop stopping: safety abort {reason}", flush=True)
                            if reason == "tumbled":
                                airborne = False
                            break
                    row = control_row(policy, adapter, shield, shield_state, latest, args, anchor_xy, now_s, previous_action, controls_drone=True)
                    anchor_xy = (float(row["target_x"]), float(row["target_y"]))
                    previous_action[:] = [row["action_thrust"], row["action_roll_rate"], row["action_pitch_rate"], row["action_yaw_rate"]]
                    motion.start_linear_motion(row["vx_m_s"], row["vy_m_s"], row["vz_m_s"], rate_yaw=row["yawrate_deg_s"])
                    rows.append(row)
        finally:
            if airborne:
                settle_before_landing(motion)
                motion.land(velocity=config.safety.velocity_m_s)
            commander.send_stop_setpoint()
            commander.send_notify_setpoint_stop()
            disarm_crazyflie_after_flight(scf.cf)
    return rows


def replay_rows(policy, adapter: SixDofVelocityAdapterConfig, shield: SixDofVelocityShieldConfig, args) -> list[dict]:
    rows = []
    previous_action = np.zeros(4, dtype=np.float32)
    anchor_xy = None
    shield_state = SixDofVelocityShieldState()
    latest: dict[str, float] = {}
    with Path(args.replay_input).open(newline="") as handle:
        for telemetry in csv.DictReader(handle):
            latest.update({key: parse_float(value) for key, value in telemetry.items() if value != ""})
            now_s = latest.get("host_time_s", len(rows) * 0.01)
            row = control_row(policy, adapter, shield, shield_state, dict(latest), args, anchor_xy, now_s, previous_action, controls_drone=False)
            anchor_xy = (float(row["target_x"]), float(row["target_y"]))
            previous_action[:] = [row["action_thrust"], row["action_roll_rate"], row["action_pitch_rate"], row["action_yaw_rate"]]
            rows.append(row)
    return rows


def control_row(policy, adapter, shield, shield_state, telemetry: dict, args, anchor_xy, now_s: float, previous_action, *, controls_drone: bool) -> dict:
    env = SixDofCrazyflieEnv(num_envs=1, seed=0, task="obstacle_avoidance")
    update_env(env, telemetry, args.height_m, anchor_xy, previous_action)
    with torch.no_grad():
        action = policy(torch.from_numpy(env.observation()).float()).cpu().numpy()[0]
    command = sixdof_action_to_velocity_command(env, action, adapter)
    shield_row, active_command = raw_policy_row(command, telemetry) if args.raw_policy_control else shield_policy_row(command, telemetry, now_s, args, shield, shield_state)
    return {
        **telemetry,
        "host_time_s": float(telemetry.get("host_time_s", time()) or time()),
        "controls_drone": controls_drone,
        "target_x": float(env.target_position[0, 0]),
        "target_y": float(env.target_position[0, 1]),
        "target_z": float(env.target_position[0, 2]),
        "action_thrust": float(action[0]),
        "action_roll_rate": float(action[1]),
        "action_pitch_rate": float(action[2]),
        "action_yaw_rate": float(action[3]),
        "puffer_vx_m_s": command.vx_m_s,
        "puffer_vy_m_s": command.vy_m_s,
        "puffer_vz_m_s": command.vz_m_s,
        "puffer_yawrate_deg_s": command.yawrate_deg_s,
        "raw_policy_control": args.raw_policy_control,
        **shield_row,
        "vx_m_s": active_command.vx_m_s,
        "vy_m_s": active_command.vy_m_s,
        "vz_m_s": active_command.vz_m_s,
        "yawrate_deg_s": active_command.yawrate_deg_s,
    }


def shield_policy_row(command, telemetry, now_s, args, shield, shield_state):
    shielded = apply_sixdof_velocity_shield(command, telemetry, now_s=now_s, target_height_m=args.height_m, config=shield, state=shield_state)
    return sixdof_shield_row(shielded), shielded.command


def raw_policy_row(command, telemetry):
    return {
        "shield_active": False,
        "min_horizontal_range_m": min(range_m(telemetry, key) for key in ("range.front", "range.back", "range.left", "range.right")),
        "min_horizontal_ttc_s": float("inf"),
    }, command


def update_env(env: SixDofCrazyflieEnv, telemetry: dict, target_z: float, anchor_xy, previous_action) -> None:
    x = value(telemetry, "stateEstimate.x")
    y = value(telemetry, "stateEstimate.y")
    env.position[0] = [x, y, value(telemetry, "stateEstimate.z", target_z)]
    env.velocity[0] = [value(telemetry, "stateEstimate.vx"), value(telemetry, "stateEstimate.vy"), value(telemetry, "stateEstimate.vz")]
    env.quaternion[0] = euler_to_quat(
        np.asarray([radians(value(telemetry, "stabilizer.roll"))]),
        np.asarray([radians(value(telemetry, "stabilizer.pitch"))]),
        np.asarray([radians(value(telemetry, "stabilizer.yaw"))]),
    )[0]
    env.body_rates[0] = [radians(value(telemetry, "gyro.x")), radians(value(telemetry, "gyro.y")), radians(value(telemetry, "gyro.z"))]
    env.ranges_m[0] = [range_m(telemetry, key) for key in ("range.front", "range.back", "range.left", "range.right", "range.up", "range.zrange")]
    target_x, target_y = anchor_xy if anchor_xy is not None else (x, y)
    env.target_position[0] = [target_x, target_y, target_z]
    env.target_yaw[0] = radians(value(telemetry, "stabilizer.yaw"))
    if previous_action is not None:
        env.previous_action[0] = previous_action


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
    }


def value(telemetry: dict, key: str, default: float = 0.0) -> float:
    return parse_float(telemetry.get(key, default))


def range_m(telemetry: dict, key: str) -> float:
    raw = value(telemetry, key, 4000.0)
    return 4.0 if raw >= 32000.0 else raw / 1000.0


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


def settle_before_landing(motion) -> None:
    motion.start_linear_motion(0.0, 0.0, 0.0, rate_yaw=0.0)
    sleep(0.8)
    motion.stop()
    sleep(0.4)


if __name__ == "__main__":
    main()
