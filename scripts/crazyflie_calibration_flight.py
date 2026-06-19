from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from time import sleep, time

from flightrl.hardware.calibration_flight import CALIBRATION_LOG_VARIABLES, build_calibration_sequence, command_row, sequence_duration_s
from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.motion import arm_crazyflie_for_flight, disarm_crazyflie_after_flight, install_legacy_hover_warning_filter, reset_crazyflie_estimator
from flightrl.hardware.preflight import require_supervisor_allows_flight
from flightrl.hardware.telemetry import build_log_configs, with_available_log_variables, with_extra_log_variables


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a known Crazyflie calibration trajectory for sim-to-real replay fitting")
    parser.add_argument("--hardware-config", default="configs/hardware/crazyflie_2_1_brushless.toml")
    parser.add_argument("--output", default="artifacts/crazyflie_logs/calibration_flight.csv")
    parser.add_argument("--pattern", default="line_yaw_square", choices=("line", "yaw", "square", "line_yaw_square"))
    parser.add_argument("--height-m", type=float, default=0.55)
    parser.add_argument("--segment-s", type=float, default=1.6)
    parser.add_argument("--hover-s", type=float, default=1.0)
    parser.add_argument("--speed-m-s", type=float, default=0.12)
    parser.add_argument("--yawrate-deg-s", type=float, default=20.0)
    parser.add_argument("--confirm-flight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sequence = build_calibration_sequence(
        pattern=args.pattern,
        segment_s=args.segment_s,
        hover_s=args.hover_s,
        speed_m_s=args.speed_m_s,
        yawrate_deg_s=args.yawrate_deg_s,
    )
    if args.dry_run:
        print(f"dry_run calibration sequence: duration_s={sequence_duration_s(sequence):.2f}")
        for command in sequence:
            print(asdict(command))
        return
    if not args.confirm_flight:
        raise SystemExit("--confirm-flight is required for real drone control")
    rows = run_live(args, sequence)
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


def run_live(args, sequence) -> list[dict[str, float | str]]:
    config = with_extra_log_variables(load_hardware_config(args.hardware_config), CALIBRATION_LOG_VARIABLES)
    modules = require_cflib()
    install_legacy_hover_warning_filter()
    rows: list[dict[str, float | str]] = []
    latest: dict[str, float] = {}
    with sync_crazyflie_context(config, modules) as scf:
        log_config = with_available_log_variables(scf, config)
        commander = scf.cf.commander
        motion = modules.motion_commander_cls(scf, default_height=args.height_m)
        airborne = False
        try:
            require_supervisor_allows_flight(scf, modules, config)
            reset_crazyflie_estimator(scf.cf)
            arm_crazyflie_for_flight(scf.cf)
            motion.take_off(height=args.height_m, velocity=config.safety.velocity_m_s)
            airborne = True
            with modules.sync_logger_cls(scf, build_log_configs(modules, log_config)) as logger:
                for command in sequence:
                    deadline = time() + command.duration_s
                    motion.start_linear_motion(command.vx_m_s, command.vy_m_s, command.vz_m_s, rate_yaw=command.yawrate_deg_s)
                    while time() < deadline:
                        _timestamp, values, _conf = next(logger)
                        latest.update({key: float(value) for key, value in values.items()})
                        rows.append({"host_time_s": time(), **latest, **command_row(command)})
                motion.stop()
        finally:
            if airborne:
                motion.stop()
                sleep(0.5)
                motion.land(velocity=config.safety.velocity_m_s)
            commander.send_stop_setpoint()
            commander.send_notify_setpoint_stop()
            disarm_crazyflie_after_flight(scf.cf)
    return rows


def write_rows(path: str | Path, rows: list[dict[str, float | str]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["host_time_s"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
