from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
from time import sleep, time

import numpy as np

from flightrl.hardware.calibration_flight import CALIBRATION_PATTERNS, CalibrationCommand, build_calibration_sequence
from flightrl.hardware.avoidance_live import next_log_sample
from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.console_capture import CrazyflieConsoleCapture
from flightrl.hardware.motion import (
    arm_crazyflie_for_flight,
    disarm_crazyflie_after_flight,
    install_legacy_hover_warning_filter,
)
from flightrl.hardware.preflight import require_supervisor_allows_flight
from flightrl.hardware.sixdof_puffer_shadow import (
    PUFFER_SHADOW_LOG_VARIABLES,
    PufferShadowConfig,
    puffer_shadow_row,
    synthetic_puffer_shadow_telemetry,
)
from flightrl.hardware.sixdof_raw_action import RawPufferActionConfig
from flightrl.hardware.telemetry import build_log_configs, with_available_log_variables, with_extra_log_variables
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy


ROOT = Path(__file__).resolve().parents[1]
LEARNED_POLICY_MONITOR_ONLY = True

REQUIRED_LIVE_FIELDS = (
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
    parser = argparse.ArgumentParser(description="Fly a conservative baseline hover while logging Puffer six-DoF shadow actions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hardware-config", default="configs/hardware/crazyflie_2_1_brushless_flow_only.toml")
    parser.add_argument("--output", default="artifacts/crazyflie_logs/baseline_puffer_shadow.csv")
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--height-m", type=float, default=0.35)
    parser.add_argument("--movement-pattern", choices=("hover", *CALIBRATION_PATTERNS), default="hover")
    parser.add_argument("--segment-s", type=float, default=0.8)
    parser.add_argument("--segment-hover-s", type=float, default=0.5)
    parser.add_argument("--speed-m-s", type=float, default=0.08)
    parser.add_argument("--yawrate-deg-s", type=float, default=15.0)
    parser.add_argument("--hover-thrust-percent", type=float, default=49.0)
    parser.add_argument("--thrust-scale", type=float, default=0.75)
    parser.add_argument("--max-roll-rate-deg-s", type=float, default=343.7747)
    parser.add_argument("--max-pitch-rate-deg-s", type=float, default=343.7747)
    parser.add_argument("--max-yaw-rate-deg-s", type=float, default=229.1831)
    parser.add_argument("--target-yaw-deg", type=float, default=0.0)
    parser.add_argument("--target-mode", choices=("current_pose", "fixed_origin"), default="current_pose")
    parser.add_argument("--previous-action-observation-scale", type=float, default=0.25)
    parser.add_argument("--log-timeout-s", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-flight", action="store_true")
    parser.add_argument("--vision-output")
    parser.add_argument("--vision-frame-dir")
    parser.add_argument("--vision-frames", type=int, default=48)
    parser.add_argument("--vision-transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--vision-host", default="192.168.4.1")
    parser.add_argument("--vision-port", type=int, default=5000)
    parser.add_argument("--vision-bind-port", type=int, default=5001)
    parser.add_argument("--vision-policy-checkpoint")
    parser.add_argument("--console-output")
    args = parser.parse_args()

    policy = load_puffer_sixdof_policy(args.checkpoint)
    if policy.metadata.observation_dim != 28 or policy.metadata.action_dim != 4:
        raise SystemExit(f"unsupported Puffer shape: obs={policy.metadata.observation_dim} action={policy.metadata.action_dim}")
    shadow_config = shadow_config_from_args(args)
    if args.dry_run:
        rows = [dry_run_row(policy, shadow_config, args)]
    else:
        if not args.confirm_flight:
            raise SystemExit("--confirm-flight is required for real baseline flight")
        rows = live_rows(policy, shadow_config, args)
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"baseline_controls_drone={not args.dry_run} puffer_controls_drone=False raw_puffer_output=True")


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


def dry_run_row(policy, config: PufferShadowConfig, args) -> dict:
    row = puffer_shadow_row(policy, synthetic_puffer_shadow_telemetry(config), config, previous_action=np.zeros(4, dtype=np.float32))
    return annotate_row(row, phase="dry_run", baseline_controls_drone=False)


def live_rows(policy, config: PufferShadowConfig, args) -> list[dict]:
    hardware = with_extra_log_variables(load_hardware_config(args.hardware_config), PUFFER_SHADOW_LOG_VARIABLES)
    modules = require_cflib()
    install_legacy_hover_warning_filter()
    latest: dict[str, float] = {}
    rows: list[dict] = []
    previous_action = np.zeros(4, dtype=np.float32)
    target = None
    vision_process: subprocess.Popen | None = None
    with sync_crazyflie_context(hardware, modules) as scf:
        console_capture = CrazyflieConsoleCapture(scf.cf, args.console_output)
        console_capture.start()
        log_config = with_available_log_variables(scf, hardware)
        commander = scf.cf.commander
        motion = modules.motion_commander_cls(scf, default_height=args.height_m)
        airborne = False
        try:
            require_supervisor_allows_flight(scf, modules, hardware)
            with modules.sync_logger_cls(scf, build_log_configs(modules, log_config)) as logger:
                require_live_telemetry(logger, latest, timeout_s=2.0, log_timeout_s=args.log_timeout_s)
                arm_crazyflie_for_flight(scf.cf)
                motion.take_off(height=args.height_m, velocity=hardware.safety.velocity_m_s)
                airborne = True
                motion.stop()
                require_started_flight(logger, latest, args.height_m, timeout_s=1.5, log_timeout_s=args.log_timeout_s)
                if args.vision_output:
                    vision_process = subprocess.Popen(vision_capture_command(args))
                    sleep(0.5)
                    if vision_process.poll() is not None:
                        raise RuntimeError("AI Deck capture failed before the movement sequence")
                for command in movement_sequence(args):
                    motion.start_linear_motion(command.vx_m_s, command.vy_m_s, command.vz_m_s, rate_yaw=command.yawrate_deg_s)
                    target = collect_shadow_rows(policy, config, args, logger, latest, previous_action, target, command, rows)
                motion.stop()
        finally:
            if airborne:
                motion.stop()
                motion.land(velocity=hardware.safety.velocity_m_s)
            commander.send_stop_setpoint()
            commander.send_notify_setpoint_stop()
            disarm_crazyflie_after_flight(scf.cf)
            finish_vision_capture(vision_process)
            console_capture.close()
    return rows


def vision_capture_command(args) -> list[str]:
    if not args.vision_frame_dir:
        raise ValueError("--vision-frame-dir is required with --vision-output")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "capture_aideck_vision.py"),
        "--transport",
        args.vision_transport,
        "--host",
        args.vision_host,
        "--port",
        str(args.vision_port),
        "--bind-port",
        str(args.vision_bind_port),
        "--frames",
        str(args.vision_frames),
        "--timeout-s",
        "3",
        "--width",
        "64",
        "--height",
        "48",
        "--include-delta",
        "--include-motion-mask",
        "--frame-dir",
        args.vision_frame_dir,
        "--output",
        args.vision_output,
    ]
    if args.vision_policy_checkpoint:
        command.extend(["--policy-checkpoint", args.vision_policy_checkpoint])
    return command


def finish_vision_capture(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=2.0)


def movement_sequence(args) -> list[CalibrationCommand]:
    if args.movement_pattern == "hover":
        return [CalibrationCommand("baseline_hover", args.duration_s)]
    return build_calibration_sequence(
        pattern=args.movement_pattern,
        segment_s=args.segment_s,
        hover_s=args.segment_hover_s,
        speed_m_s=args.speed_m_s,
        yawrate_deg_s=args.yawrate_deg_s,
    )


def collect_shadow_rows(
    policy,
    config: PufferShadowConfig,
    args,
    logger,
    latest: dict[str, float],
    previous_action: np.ndarray,
    target: np.ndarray | None,
    command: CalibrationCommand,
    rows: list[dict],
) -> np.ndarray | None:
    deadline = time() + command.duration_s
    while time() < deadline:
        sample = next_log_sample(logger, timeout_s=args.log_timeout_s)
        if sample is None:
            print(f"baseline shadow stopping during {command.mode}: log timeout", flush=True)
            break
        _timestamp, values, _conf = sample
        latest.update({key: float(value) for key, value in values.items()})
        latest["host_time_s"] = time()
        if not has_required_live_telemetry(latest):
            continue
        if target is None:
            target = target_from_row(latest, args.target_mode, args.height_m)
        row = puffer_shadow_row(policy, latest, config, previous_action=previous_action, target=target)
        previous_action[:] = [row["action_thrust"], row["action_roll_rate"], row["action_pitch_rate"], row["action_yaw_rate"]]
        rows.append(annotate_row(row, command=command, baseline_controls_drone=True))
    return target


def require_live_telemetry(logger, latest: dict[str, float], *, timeout_s: float, log_timeout_s: float) -> None:
    deadline = time() + timeout_s
    while time() < deadline:
        sample = next_log_sample(logger, timeout_s=log_timeout_s)
        if sample is None:
            continue
        _timestamp, values, _conf = sample
        latest.update({key: float(value) for key, value in values.items()})
        latest["host_time_s"] = time()
        if has_required_live_telemetry(latest) and has_plausible_live_telemetry(latest):
            return
    missing = [key for key in REQUIRED_LIVE_FIELDS if key not in latest]
    raise RuntimeError(f"live telemetry did not become usable before arming; missing={missing}")


def require_started_flight(logger, latest: dict[str, float], height_m: float, *, timeout_s: float, log_timeout_s: float) -> None:
    deadline = time() + timeout_s
    min_started_z = min(0.15, max(0.08, 0.6 * height_m))
    evidence_streak = 0
    while time() < deadline:
        sample = next_log_sample(logger, timeout_s=log_timeout_s)
        if sample is None:
            continue
        _timestamp, values, _conf = sample
        latest.update({key: float(value) for key, value in values.items()})
        latest["host_time_s"] = time()
        if has_takeoff_evidence(latest, min_started_z):
            evidence_streak += 1
        else:
            evidence_streak = 0
        if evidence_streak >= 3:
            return
    raise RuntimeError(
        "baseline takeoff did not start; measured altitude did not rise despite the commanded takeoff"
    )


def has_takeoff_evidence(values: dict[str, float], min_started_z: float) -> bool:
    floor_range_mm = values.get("range.zrange")
    floor_clearance_ok = floor_range_mm is None or float(floor_range_mm) >= 800.0 * min_started_z
    return (
        float(values.get("sys.isFlying", 0.0)) > 0.0
        and float(values.get("stateEstimate.z", 0.0)) >= min_started_z
        and floor_clearance_ok
        and abs(float(values.get("stabilizer.roll", 0.0))) <= 12.0
        and abs(float(values.get("stabilizer.pitch", 0.0))) <= 12.0
    )


def has_required_live_telemetry(values: dict[str, float]) -> bool:
    return all(key in values for key in REQUIRED_LIVE_FIELDS)


def has_plausible_live_telemetry(values: dict[str, float]) -> bool:
    accel_norm = abs(float(values.get("acc.x", 0.0))) + abs(float(values.get("acc.y", 0.0))) + abs(float(values.get("acc.z", 0.0)))
    return float(values.get("pm.vbat", 0.0)) > 3.0 and accel_norm > 0.05


def target_from_row(row: dict[str, float], mode: str, height_m: float) -> np.ndarray:
    if mode == "fixed_origin":
        return np.asarray([0.0, 0.0, height_m], dtype=np.float32)
    return np.asarray([float(row.get("stateEstimate.x", 0.0)), float(row.get("stateEstimate.y", 0.0)), height_m], dtype=np.float32)


def annotate_row(
    row: dict,
    *,
    command: CalibrationCommand | None = None,
    phase: str | None = None,
    baseline_controls_drone: bool,
) -> dict:
    command = command or CalibrationCommand(phase or "baseline_hover", 0.0)
    return {
        **row,
        "phase": command.mode,
        "baseline_vx_m_s": command.vx_m_s,
        "baseline_vy_m_s": command.vy_m_s,
        "baseline_vz_m_s": command.vz_m_s,
        "baseline_yawrate_deg_s": command.yawrate_deg_s,
        "baseline_controls_drone": baseline_controls_drone,
        "puffer_controls_drone": False,
    }


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
