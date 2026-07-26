from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import time

import numpy as np

from flightrl.hardware.avoidance_live import next_log_sample
from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.sixdof_puffer_shadow import (
    PUFFER_SHADOW_LOG_VARIABLES,
    PufferShadowConfig,
    puffer_shadow_row,
    synthetic_puffer_shadow_telemetry,
    write_rows,
)
from flightrl.hardware.sixdof_raw_action import RawPufferActionConfig
from flightrl.hardware.telemetry import build_log_configs, with_available_log_variables, with_extra_log_variables
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor raw Puffer six-DoF actions from live Crazyflie telemetry without control.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hardware-config", default="configs/hardware/crazyflie_2_1_brushless.toml")
    parser.add_argument("--output", default="artifacts/crazyflie_logs/sixdof_puffer_shadow_monitor.csv")
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--height-m", type=float, default=0.50)
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
    args = parser.parse_args()

    policy = load_puffer_sixdof_policy(args.checkpoint)
    if policy.metadata.observation_dim != 28 or policy.metadata.action_dim != 4:
        raise SystemExit(f"unsupported Puffer shape: obs={policy.metadata.observation_dim} action={policy.metadata.action_dim}")
    config = shadow_config_from_args(args)
    if args.dry_run:
        rows = [puffer_shadow_row(policy, synthetic_puffer_shadow_telemetry(config), config)]
    elif args.replay_input:
        rows = replay_rows(policy, config, args.replay_input, args.target_mode)
    else:
        rows = live_rows(policy, config, args)
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print("monitor_only=True controls_drone=False raw_puffer_output=True")


def shadow_config_from_args(args) -> PufferShadowConfig:
    return PufferShadowConfig(
        height_m=args.height_m,
        target_yaw_deg=args.target_yaw_deg,
        previous_action_observation_scale=args.previous_action_observation_scale,
        raw_action=RawPufferActionConfig(
            hover_thrust_percent=args.hover_thrust_percent,
            thrust_scale=args.thrust_scale,
            max_roll_rate_deg_s=args.max_roll_rate_deg_s,
            max_pitch_rate_deg_s=args.max_pitch_rate_deg_s,
            max_yaw_rate_deg_s=args.max_yaw_rate_deg_s,
        ),
    )


def live_rows(policy, config: PufferShadowConfig, args) -> list[dict]:
    hardware = with_extra_log_variables(load_hardware_config(args.hardware_config), PUFFER_SHADOW_LOG_VARIABLES)
    modules = require_cflib()
    latest: dict[str, float] = {}
    rows: list[dict] = []
    previous_action = np.zeros(4, dtype=np.float32)
    target = None
    deadline = time() + args.duration_s
    with sync_crazyflie_context(hardware, modules) as scf:
        log_config = with_available_log_variables(scf, hardware)
        with modules.sync_logger_cls(scf, build_log_configs(modules, log_config)) as logger:
            while time() < deadline:
                sample = next_log_sample(logger, timeout_s=args.log_timeout_s)
                if sample is None:
                    print("puffer shadow stopping: log timeout", flush=True)
                    break
                _timestamp, values, _conf = sample
                latest.update({key: float(value) for key, value in values.items()})
                latest["host_time_s"] = time()
                if target is None:
                    target = target_from_row(latest, args.target_mode, config.height_m)
                row = puffer_shadow_row(policy, latest, config, previous_action=previous_action, target=target)
                previous_action[:] = [row["action_thrust"], row["action_roll_rate"], row["action_pitch_rate"], row["action_yaw_rate"]]
                rows.append(row)
    return rows


def replay_rows(policy, config: PufferShadowConfig, replay_input: str, target_mode: str) -> list[dict]:
    latest: dict[str, float] = {}
    rows: list[dict] = []
    previous_action = np.zeros(4, dtype=np.float32)
    target = None
    with Path(replay_input).open(newline="") as handle:
        for telemetry in csv.DictReader(handle):
            latest.update({key: parse_float(value) for key, value in telemetry.items() if value != ""})
            if target is None:
                target = target_from_row(latest, target_mode, config.height_m)
            row = puffer_shadow_row(policy, dict(latest), config, previous_action=previous_action, target=target)
            previous_action[:] = [row["action_thrust"], row["action_roll_rate"], row["action_pitch_rate"], row["action_yaw_rate"]]
            rows.append(row)
    return rows


def target_from_row(row: dict[str, float], mode: str, height_m: float) -> np.ndarray:
    if mode == "fixed_origin":
        return np.asarray([0.0, 0.0, height_m], dtype=np.float32)
    return np.asarray([float(row.get("stateEstimate.x", 0.0)), float(row.get("stateEstimate.y", 0.0)), height_m], dtype=np.float32)


def parse_float(raw) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
