from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from time import sleep, time

from flightrl.hardware.avoidance_policy import reading_from_telemetry, vertical_velocity_from_height_error
from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.motion import (
    arm_crazyflie_for_flight,
    disarm_crazyflie_after_flight,
    install_legacy_hover_warning_filter,
)
from flightrl.hardware.preflight import require_supervisor_allows_flight
from flightrl.hardware.telemetry import build_log_configs, with_available_log_variables, with_extra_log_variables


SCAN_LOG_VARIABLES = (
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "range.zrange",
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "pm.vbat",
)


@dataclass(frozen=True, slots=True)
class ScanCommand:
    vx_m_s: float
    vy_m_s: float
    vz_m_s: float
    yawrate_deg_s: float
    mode: str


@dataclass(frozen=True, slots=True)
class HeightTarget:
    zdistance_m: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore a room with bounded Crazyflie ranger motion and telemetry logging")
    parser.add_argument("--hardware-config", default="configs/hardware/crazyflie_2_1_brushless.toml")
    parser.add_argument("--output", default="artifacts/crazyflie_logs/room_scan.csv")
    parser.add_argument("--duration-s", type=float, default=35.0)
    parser.add_argument("--height-m", type=float, default=0.55)
    parser.add_argument("--clearance-m", type=float, default=0.55)
    parser.add_argument("--hard-clearance-m", type=float, default=0.22)
    parser.add_argument("--max-speed-m-s", type=float, default=0.14)
    parser.add_argument("--max-vertical-speed-m-s", type=float, default=0.15)
    parser.add_argument("--yawrate-deg-s", type=float, default=18.0)
    parser.add_argument("--min-airborne-height-m", type=float, default=0.20)
    parser.add_argument("--takeoff-timeout-s", type=float, default=5.0)
    parser.add_argument("--confirm-flight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.confirm_flight and not args.dry_run:
        raise SystemExit("--confirm-flight is required for real drone control")
    if args.dry_run:
        reading = reading_from_telemetry({"range.front": 1800.0, "range.left": 900.0, "range.zrange": args.height_m * 1000.0})
        print(f"dry_run scan command: {build_scan_command(reading, args)}")
        return
    rows = run_live(args)
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


def run_live(args) -> list[dict[str, float | str]]:
    config = with_extra_log_variables(load_hardware_config(args.hardware_config), SCAN_LOG_VARIABLES)
    modules = require_cflib()
    install_legacy_hover_warning_filter()
    latest: dict[str, float] = {}
    rows: list[dict[str, float | str]] = []
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
            print(f"room scan started: duration_s={args.duration_s:.1f}, height_m={args.height_m:.2f}", flush=True)
            deadline = time() + args.duration_s
            with modules.sync_logger_cls(scf, build_log_configs(modules, log_config)) as logger:
                if not wait_until_airborne(logger, latest, rows, args):
                    print(
                        "room scan abort: Crazyflie did not reach min airborne height "
                        f"{args.min_airborne_height_m:.2f} m",
                        flush=True,
                    )
                    return [{"host_time_s": time(), **latest, "mode": "abort_not_airborne"}]
                while time() < deadline:
                    values = read_next_values(logger)
                    if values is None:
                        break
                    latest.update(values)
                    command = build_scan_command(reading_from_telemetry(latest), args)
                    motion.start_linear_motion(command.vx_m_s, command.vy_m_s, command.vz_m_s, rate_yaw=command.yawrate_deg_s)
                    rows.append({"host_time_s": time(), **latest, **asdict(command)})
        finally:
            if airborne:
                motion.stop()
                sleep(0.5)
                motion.land(velocity=config.safety.velocity_m_s)
            commander.send_stop_setpoint()
            commander.send_notify_setpoint_stop()
            disarm_crazyflie_after_flight(scf.cf)
    return rows


def read_next_values(logger) -> dict[str, float] | None:
    try:
        _timestamp, values, _conf = next(logger)
    except StopIteration:
        return None
    return {key: float(value) for key, value in values.items()}


def wait_until_airborne(logger, latest: dict[str, float], rows: list[dict[str, float | str]], args) -> bool:
    deadline = time() + args.takeoff_timeout_s
    while time() < deadline:
        values = read_next_values(logger)
        if values is None:
            return False
        latest.update(values)
        rows.append({"host_time_s": time(), **latest, "vx_m_s": 0.0, "vy_m_s": 0.0, "vz_m_s": 0.0, "yawrate_deg_s": 0.0, "mode": "takeoff_wait"})
        if float(latest.get("stateEstimate.z", 0.0)) >= args.min_airborne_height_m:
            return True
    return False


def build_scan_command(reading, args) -> ScanCommand:
    lateral = {
        "front": reading.front_m,
        "back": reading.back_m,
        "left": reading.left_m,
        "right": reading.right_m,
    }
    closest = min(lateral, key=lateral.get)
    if lateral[closest] <= args.hard_clearance_m:
        vx, vy = escape_velocity(closest, args.max_speed_m_s)
        mode = f"escape_{closest}"
    else:
        direction = max(lateral, key=lateral.get)
        speed = exploration_speed(lateral[direction], args.clearance_m, args.max_speed_m_s)
        vx, vy = direction_velocity(direction, speed)
        mode = f"explore_{direction}" if speed > 0.0 else "yaw_scan"
    vertical = vertical_velocity_from_height_error(
        HeightTarget(args.height_m),
        reading,
        max_vertical_speed_m_s=args.max_vertical_speed_m_s,
    )
    return ScanCommand(vx, vy, vertical, args.yawrate_deg_s, mode)


def direction_velocity(direction: str, speed: float) -> tuple[float, float]:
    if direction == "front":
        return speed, 0.0
    if direction == "back":
        return -speed, 0.0
    if direction == "left":
        return 0.0, speed
    return 0.0, -speed


def escape_velocity(closest: str, speed: float) -> tuple[float, float]:
    opposite = {"front": "back", "back": "front", "left": "right", "right": "left"}[closest]
    return direction_velocity(opposite, speed)


def exploration_speed(distance_m: float, clearance_m: float, max_speed_m_s: float) -> float:
    if distance_m <= clearance_m:
        return 0.0
    return float(min(max_speed_m_s, max_speed_m_s * (distance_m - clearance_m) / clearance_m))


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
