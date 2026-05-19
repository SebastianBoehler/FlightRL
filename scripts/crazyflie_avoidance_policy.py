from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import sleep, time

import torch

from flightrl.hardware.avoidance_policy import RangerAvoidancePolicy, command_from_model, command_row, reading_from_telemetry
from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.motion import arm_for_flight, disarm_after_flight
from flightrl.hardware.telemetry import build_log_configs


AVOIDANCE_LOG_VARIABLES = ("range.front", "range.back", "range.left", "range.right", "range.up", "range.zrange", "pm.vbat")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained ranger avoidance policy on Crazyflie hover setpoints")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hardware-config", default="configs/hardware/crazyflie_2_1_brushless.toml")
    parser.add_argument("--output", default="artifacts/crazyflie_logs/avoidance_policy.csv")
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--height-m", type=float, default=0.45)
    parser.add_argument("--confirm-flight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model = load_policy(args.checkpoint)
    if args.dry_run:
        reading = reading_from_telemetry({"range.front": 250.0, "range.back": 2000.0, "range.zrange": args.height_m * 1000.0})
        command = command_from_model(model, reading)
        print(f"dry_run avoidance command: {command}")
        return
    if not args.confirm_flight:
        raise SystemExit("--confirm-flight is required for real drone control")
    rows = run_live(model, args)
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


def load_policy(path: str | Path) -> RangerAvoidancePolicy:
    checkpoint = torch.load(path, map_location="cpu")
    hidden_size = int(checkpoint.get("hidden_size", 64))
    model = RangerAvoidancePolicy(hidden_size=hidden_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def run_live(model: RangerAvoidancePolicy, args) -> list[dict[str, float]]:
    config = load_hardware_config(args.hardware_config)
    config.logging.variables = AVOIDANCE_LOG_VARIABLES
    modules = require_cflib()
    latest: dict[str, float] = {}
    rows: list[dict[str, float]] = []
    deadline = time() + args.duration_s
    with sync_crazyflie_context(config, modules) as scf:
        commander = scf.cf.commander
        try:
            arm_for_flight(scf.cf.supervisor)
            commander.send_hover_setpoint(0.0, 0.0, 0.0, args.height_m)
            sleep(2.0)
            with modules.sync_logger_cls(scf, build_log_configs(modules, config)) as logger:
                while time() < deadline:
                    _timestamp, values, _conf = next(logger)
                    latest.update({key: float(value) for key, value in values.items()})
                    command = command_from_model(model, reading_from_telemetry(latest))
                    commander.send_hover_setpoint(command.vx_m_s, command.vy_m_s, command.yawrate_deg_s, command.zdistance_m)
                    rows.append({"host_time_s": time(), **latest, **command_row(command)})
        finally:
            commander.send_hover_setpoint(0.0, 0.0, 0.0, args.height_m)
            sleep(0.5)
            commander.send_stop_setpoint()
            commander.send_notify_setpoint_stop()
            disarm_after_flight(scf.cf.supervisor)
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
