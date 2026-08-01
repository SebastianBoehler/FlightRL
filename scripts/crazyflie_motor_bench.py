from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from time import monotonic, sleep

from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.errors import HardwareError, HardwareSafetyError
from flightrl.hardware.motion import arm_crazyflie_for_flight, disarm_crazyflie_after_flight
from flightrl.hardware.preflight import require_supervisor_allows_flight, require_supervisor_is_armed_and_can_fly
from flightrl.hardware.telemetry import next_log_packet


MAX_BENCH_POWER = 32_000
MAX_BENCH_STEPS = 32
MOTOR_LOG_WATCHDOG_S = 0.2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prop-off Crazyflie motor output bench")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="explicit hardware/deck profile for the connected stack",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/crazyflie_logs/motor_bench.csv"))
    parser.add_argument("--powers", type=int, nargs="+", default=[14000, 20000, 26000, 32000])
    parser.add_argument("--motors", type=int, nargs="+", default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    parser.add_argument("--confirm-props-off", action="store_true", help="required for real motor output")
    parser.add_argument("--dry-run", action="store_true", help="print planned sequence without touching cflib")
    args = parser.parse_args(argv)

    try:
        validate_bench_plan(args.powers, args.motors)
    except ValueError as exc:
        parser.error(str(exc))
    config = load_hardware_config(args.config)
    if args.dry_run:
        print(f"dry_run motor bench: uri={config.radio.uri}")
        for motor in args.motors:
            print(f"m{motor}: powers={args.powers}")
        return 0
    if not args.confirm_props_off:
        parser.error("--confirm-props-off is required for real motor output")

    try:
        rows = run_bench(config, args.powers, args.motors)
    except HardwareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "motor",
                "power",
                "rpm",
                "motor_rpm",
                "motor_output",
                "motor_requested",
                "vbat",
                "supervisor_info",
                "motor_pass",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


def run_bench(config, powers: list[int], motors: list[int]) -> list[dict[str, object]]:
    validate_bench_plan(powers, motors)
    modules = require_cflib()
    rows: list[dict[str, object]] = []
    with sync_crazyflie_context(config, modules) as scf:
        cf = scf.cf
        try:
            _zero_all(cf)
            require_supervisor_allows_flight(scf, modules, config)
            arm_crazyflie_for_flight(cf)
            require_supervisor_is_armed_and_can_fly(scf, modules, config)
            _set_param(cf, "motorPowerSet.enable", 1)
            for motor in motors:
                rows.extend(_run_motor(cf, scf, modules, motor, powers))
        finally:
            _zero_all(cf)
            cf.commander.send_stop_setpoint()
            cf.commander.send_notify_setpoint_stop()
            disarm_crazyflie_after_flight(cf)
    return rows


def _run_motor(cf, scf, modules, motor: int, powers: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    logconf = modules.log_config_cls(name=f"Motor{motor}", period_in_ms=20)
    variables = _available_variables(
        cf,
        [
            f"rpm.m{motor}",
            f"motor.m{motor}_rpm",
            f"motor.m{motor}",
            f"motor.m{motor}req",
            "pm.vbat",
            "supervisor.info",
            "health.motorPass",
        ],
    )
    if not variables:
        raise RuntimeError("no motor bench log variables are available in the Crazyflie TOC")
    print(f"m{motor}: logging {variables}")
    for variable in variables:
        logconf.add_variable(variable)
    with modules.sync_logger_cls(scf, logconf) as logger:
        for power in powers:
            arm_crazyflie_for_flight(cf)
            _set_param(cf, f"motorPowerSet.m{motor}", power)
            try:
                latest = _collect_latest(logger, 0.45)
            finally:
                _set_param(cf, f"motorPowerSet.m{motor}", 0)
            rpm = latest.get(f"rpm.m{motor}") or latest.get(f"motor.m{motor}_rpm", "")
            row = {
                "motor": motor,
                "power": power,
                "rpm": rpm,
                "motor_rpm": latest.get(f"motor.m{motor}_rpm", ""),
                "motor_output": latest.get(f"motor.m{motor}", ""),
                "motor_requested": latest.get(f"motor.m{motor}req", ""),
                "vbat": latest.get("pm.vbat", ""),
                "supervisor_info": latest.get("supervisor.info", ""),
                "motor_pass": latest.get("health.motorPass", ""),
            }
            rows.append(row)
            print(row)
            _collect_latest(logger, 0.2)
    return rows


def _collect_latest(logger, seconds: float) -> dict:
    latest = {}
    deadline = monotonic() + seconds
    watchdog_deadline = monotonic() + MOTOR_LOG_WATCHDOG_S
    while (remaining := deadline - monotonic()) > 0.0:
        watchdog_remaining = watchdog_deadline - monotonic()
        if watchdog_remaining <= 0.0:
            raise HardwareSafetyError("motor bench telemetry watchdog expired")
        packet = next_log_packet(logger, timeout_s=min(remaining, watchdog_remaining))
        if packet is None:
            if monotonic() >= deadline:
                break
            raise HardwareSafetyError("motor bench telemetry watchdog expired")
        _timestamp, latest, _config = packet
        watchdog_deadline = monotonic() + MOTOR_LOG_WATCHDOG_S
    return dict(latest)


def validate_bench_plan(powers: list[int], motors: list[int]) -> None:
    if not powers or any(type(power) is not int or not 1 <= power <= MAX_BENCH_POWER for power in powers):
        raise ValueError(f"powers must be integers in [1, {MAX_BENCH_POWER}]")
    if not motors or any(type(motor) is not int or motor not in {1, 2, 3, 4} for motor in motors):
        raise ValueError("motors must be unique integers in [1, 4]")
    if len(set(motors)) != len(motors):
        raise ValueError("motors must be unique integers in [1, 4]")
    if len(set(powers)) != len(powers):
        raise ValueError("powers must not contain duplicates")
    if len(powers) * len(motors) > MAX_BENCH_STEPS:
        raise ValueError(f"motor bench is limited to {MAX_BENCH_STEPS} powered steps")


def _set_param(cf, name: str, value: int) -> None:
    cf.param.set_value(name, str(value))
    sleep(0.05)


def _available_variables(cf, variables: list[str]) -> list[str]:
    toc = getattr(getattr(getattr(cf, "log", None), "toc", None), "toc", None)
    if not isinstance(toc, dict):
        return variables
    available = []
    for variable in variables:
        group, _, name = variable.partition(".")
        if group in toc and name in toc[group]:
            available.append(variable)
        else:
            print(f"skipping unavailable log variable: {variable}")
    return available


def _zero_all(cf) -> None:
    for motor in range(1, 5):
        cf.param.set_value(f"motorPowerSet.m{motor}", "0")
        sleep(0.02)
    cf.param.set_value("motorPowerSet.enable", "0")
    sleep(0.08)


if __name__ == "__main__":
    raise SystemExit(main())
