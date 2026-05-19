from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flightrl.hardware.cflib_bridge import require_cflib, request_platform_info, scan_interfaces, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.errors import HardwareError
from flightrl.hardware.motion import DemoFlightPlan, build_motion_commander, execute_demo_flight
from flightrl.hardware.preflight import expected_deck_params, inspect_decks, inspect_log_variables


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "hardware" / "crazyflie_2_1_brushless.toml"


class DryRunCommander:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def take_off(self, height: float, velocity: float) -> None:
        self.calls.append(f"take_off height={height:.2f} velocity={velocity:.2f}")

    def stop(self) -> None:
        self.calls.append("stop")

    def turn_left(self, angle_degrees: float, rate: float) -> None:
        self.calls.append(f"turn_left angle={angle_degrees:.1f} rate={rate:.1f}")

    def turn_right(self, angle_degrees: float, rate: float) -> None:
        self.calls.append(f"turn_right angle={angle_degrees:.1f} rate={rate:.1f}")

    def land(self, velocity: float) -> None:
        self.calls.append(f"land velocity={velocity:.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crazyflie hardware bring-up helpers")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="validate command flow without importing cflib")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("scan", help="scan for Crazyflie radio URIs")
    subparsers.add_parser("check", help="connect and inspect expected decks")
    demo = subparsers.add_parser("demo", help="run a conservative hover/turn/land demo")
    demo.add_argument("--confirm", action="store_true", help="required before spinning motors")

    args = parser.parse_args(argv)
    config = load_hardware_config(args.config)
    try:
        if args.command == "scan":
            return _scan(args.dry_run)
        if args.command == "check":
            return _check(config, args.dry_run)
        if args.command == "demo":
            return _demo(config, args.dry_run, args.confirm)
    except HardwareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


def _scan(dry_run: bool) -> int:
    if dry_run:
        print("dry_run scan: cflib was not imported and no radio scan was performed")
        return 0
    uris = scan_interfaces()
    if not uris:
        print("no Crazyflie interfaces found")
        return 1
    for uri in uris:
        print(uri)
    return 0


def _check(config, dry_run: bool) -> int:
    if dry_run:
        print(f"dry_run check: uri={config.radio.uri}")
        print("expected deck params: " + ", ".join(expected_deck_params(config)))
        print(f"log variables: {len(config.logging.variables)}")
        return 0
    modules = require_cflib()
    with sync_crazyflie_context(config, modules) as scf:
        platform = request_platform_info(scf.cf)
        deck_report = inspect_decks(scf, config)
        log_report = inspect_log_variables(scf, config)
    for name, value in {**platform, **deck_report.details, **log_report.details}.items():
        print(f"{name}={value}")
    for warning in (*deck_report.warnings, *log_report.warnings):
        print(f"warning: {warning}", file=sys.stderr)
    return 0 if deck_report.ok and log_report.ok else 1


def _demo(config, dry_run: bool, confirmed: bool) -> int:
    plan = DemoFlightPlan.from_config(config, confirmed=confirmed or dry_run)
    if dry_run:
        commander = DryRunCommander()
        execute_demo_flight(commander, plan, sleep=lambda _: None)
        print("dry_run demo command sequence:")
        for call in commander.calls:
            print(call)
        return 0
    modules = require_cflib()
    with sync_crazyflie_context(config, modules) as scf:
        commander = build_motion_commander(scf, modules, config)
        execute_demo_flight(commander, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
