from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import time, time_ns

from flightrl.hardware.cflib_bridge import require_cflib, request_platform_info, scan_interfaces, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.errors import HardwareError, HardwareSafetyError
from flightrl.hardware.flight_telemetry import FlightTelemetryRecorder, watchdog_sleep
from flightrl.hardware.flight_validation import (
    validate_instrumented_patrol,
    validate_out_and_back,
)
from flightrl.hardware.motion import (
    DemoFlightPlan,
    PatrolFlightPlan,
    arm_crazyflie_for_flight,
    build_motion_commander,
    disarm_crazyflie_after_flight,
    execute_demo_flight,
    execute_patrol_flight,
)
from flightrl.hardware.out_and_back import (
    OutAndBackFlightPlan,
    execute_out_and_back,
)
from flightrl.hardware.preflight import (
    expected_deck_params,
    inspect_decks,
    inspect_log_variables,
    require_expected_decks,
    require_supervisor_allows_flight,
    require_supervisor_is_armed_and_can_fly,
)


class DryRunCommander:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def take_off(self, height: float, velocity: float) -> None:
        self.calls.append(f"take_off height={height:.2f} velocity={velocity:.2f}")

    def stop(self) -> None:
        self.calls.append("stop")

    def start_linear_motion(
        self,
        velocity_x_m: float,
        velocity_y_m: float,
        velocity_z_m: float,
        rate_yaw: float = 0.0,
    ) -> None:
        self.calls.append(
            "start_linear_motion "
            f"x={velocity_x_m:.2f} y={velocity_y_m:.2f} "
            f"z={velocity_z_m:.2f} yaw={rate_yaw:.1f}"
        )

    def turn_left(self, angle_degrees: float, rate: float) -> None:
        self.calls.append(f"turn_left angle={angle_degrees:.1f} rate={rate:.1f}")

    def start_turn_left(self, rate: float) -> None:
        self.calls.append(f"start_turn_left rate={rate:.1f}")

    def turn_right(self, angle_degrees: float, rate: float) -> None:
        self.calls.append(f"turn_right angle={angle_degrees:.1f} rate={rate:.1f}")

    def land(self, velocity: float) -> None:
        self.calls.append(f"land velocity={velocity:.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crazyflie hardware bring-up helpers")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="explicit hardware/deck profile for the connected stack",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate command flow without importing cflib")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("scan", help="scan for Crazyflie radio URIs")
    subparsers.add_parser("check", help="connect and inspect expected decks")
    demo = subparsers.add_parser("demo", help="run a conservative hover/turn/land demo")
    demo.add_argument("--confirm", action="store_true", help="required before spinning motors")
    patrol = subparsers.add_parser(
        "patrol",
        help="run the fixed short-forward/slow-turn/short-forward demo",
    )
    patrol.add_argument("--confirm", action="store_true", help="required before spinning motors")
    out_and_back = subparsers.add_parser(
        "out-and-back",
        help="run the fixed forward/hover/backward diagnostic",
    )
    out_and_back.add_argument("--confirm", action="store_true", help="required before spinning motors")

    args = parser.parse_args(argv)
    config = load_hardware_config(args.config)
    try:
        if args.command == "scan":
            return _scan(args.dry_run)
        if args.command == "check":
            return _check(config, args.dry_run)
        if args.command == "demo":
            return _demo(config, args.dry_run, args.confirm)
        if args.command == "patrol":
            return _patrol(config, args.dry_run, args.confirm)
        if args.command == "out-and-back":
            return _patrol(
                config,
                args.dry_run,
                args.confirm,
                behavior="out-and-back",
            )
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
        try:
            require_expected_decks(scf, config)
            require_supervisor_allows_flight(scf, modules, config)
            arm_crazyflie_for_flight(scf.cf)
            require_supervisor_is_armed_and_can_fly(scf, modules, config)
            commander = build_motion_commander(scf, modules, config)
            execute_demo_flight(commander, plan)
        finally:
            disarm_crazyflie_after_flight(scf.cf)
    return 0


def _patrol(
    config,
    dry_run: bool,
    confirmed: bool,
    *,
    behavior: str = "patrol",
) -> int:
    if config.safety.requires_manual_confirm and not (confirmed or dry_run):
        raise HardwareError("manual confirmation is required before spinning motors")
    if behavior == "out-and-back":
        plan = OutAndBackFlightPlan()
        execute = execute_out_and_back
        validate = validate_out_and_back
    else:
        plan = PatrolFlightPlan()
        execute = execute_patrol_flight
        validate = validate_instrumented_patrol
    if dry_run:
        commander = DryRunCommander()
        execute(commander, plan, sleep=lambda _: None)
        print(f"dry_run {behavior} command sequence:")
        for call in commander.calls:
            print(call)
        print(
            f"nominal_duration_s={plan.nominal_duration_s():.2f} "
            f"max_flight_s={plan.max_flight_s:.2f}"
        )
        return 0
    modules = require_cflib()
    run_dir = _flight_output_dir()
    with (run_dir / "events.jsonl").open("x") as event_log:
        def record_phase(phase: str) -> None:
            event_log.write(json.dumps({"host_time_s": time(), "phase": phase}) + "\n")
            event_log.flush()

        with sync_crazyflie_context(config, modules) as scf:
            recorder = FlightTelemetryRecorder(
                scf,
                modules,
                config,
                run_dir / "telemetry.csv",
            )
            recorder.start()
            try:
                recorder.wait_ready(timeout_s=2.0)
                recorder.require_safe(maximum_age_s=0.25)
                require_expected_decks(scf, config)
                require_supervisor_allows_flight(scf, modules, config)
                arm_crazyflie_for_flight(scf.cf)
                require_supervisor_is_armed_and_can_fly(scf, modules, config)
                commander = build_motion_commander(scf, modules, config)
                execute(
                    commander,
                    plan,
                    sleep=lambda duration_s: watchdog_sleep(
                        duration_s,
                        recorder=recorder,
                        maximum_age_s=0.25,
                    ),
                    on_phase=record_phase,
                )
            finally:
                try:
                    disarm_crazyflie_after_flight(scf.cf)
                finally:
                    recorder.close()
    validation = validate(run_dir)
    (run_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    print(f"patrol telemetry: {run_dir / 'telemetry.csv'}")
    print(f"patrol telemetry samples: {recorder.sample_count}")
    passed_key = (
        "out_and_back_passed"
        if behavior == "out-and-back"
        else "instrumented_patrol_passed"
    )
    print(f"instrumented {behavior} passed: {validation[passed_key]}")
    if not validation[passed_key]:
        failures = ", ".join(validation["failed_checks"])
        raise HardwareSafetyError(f"instrumented {behavior} failed: {failures}")
    return 0


def _flight_output_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "crazyflie_flights" / f"patrol_{time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
