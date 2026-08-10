from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config, validate_hardware_config
from flightrl.hardware.console_capture import CrazyflieConsoleCapture
from flightrl.hardware.errors import HardwareError
from flightrl.hardware.telemetry import default_log_path, validate_log_duration, write_sync_log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record Crazyflie telemetry to replay-friendly CSV")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="explicit hardware/deck profile for the connected stack",
    )
    parser.add_argument("--uri", help="override the configured read-only telemetry URI")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--console-output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate config without recording telemetry")
    args = parser.parse_args(argv)

    try:
        validate_log_duration(args.duration_s)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        config = load_hardware_config(args.config)
        if args.uri is not None:
            config = replace(config, radio=replace(config.radio, uri=args.uri))
            validate_hardware_config(config)
        output = args.output or default_log_path(config)
        if args.dry_run:
            print(f"dry_run log: uri={config.radio.uri}")
            print(f"dry_run log: output={output}")
            print("dry_run log: no telemetry was recorded")
            return 0
        modules = require_cflib()
        with sync_crazyflie_context(config, modules) as scf:
            console_capture = CrazyflieConsoleCapture(scf.cf, args.console_output)
            console_capture.start()
            try:
                count = write_sync_log(scf, modules, config, output, args.duration_s)
            finally:
                console_capture.close()
        print(f"wrote {count} telemetry samples to {output}")
        return 0
    except HardwareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
