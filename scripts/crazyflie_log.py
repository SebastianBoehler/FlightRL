from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.errors import HardwareError
from flightrl.hardware.telemetry import default_log_path, write_sync_log


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "hardware" / "crazyflie_2_1_brushless.toml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record Crazyflie telemetry to replay-friendly CSV")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate config without recording telemetry")
    args = parser.parse_args(argv)

    try:
        config = load_hardware_config(args.config)
        output = args.output or default_log_path(config)
        if args.dry_run:
            print(f"dry_run log: uri={config.radio.uri}")
            print(f"dry_run log: output={output}")
            print("dry_run log: no telemetry was recorded")
            return 0
        modules = require_cflib()
        with sync_crazyflie_context(config, modules) as scf:
            count = write_sync_log(scf, modules, config, output, args.duration_s)
        print(f"wrote {count} telemetry samples to {output}")
        return 0
    except HardwareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
