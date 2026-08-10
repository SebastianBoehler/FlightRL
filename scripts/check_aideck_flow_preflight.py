from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.errors import HardwareError
from flightrl.hardware.flow_preflight_contract import (
    inspect_exact_flow_preflight_stack,
    validate_flow_preflight_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only exact AI Deck, Flow Deck, Z-ranger, and TOC check."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_hardware_config(args.config)
        validate_flow_preflight_config(config)
        modules = require_cflib()
        with sync_crazyflie_context(config, modules) as scf:
            report = inspect_exact_flow_preflight_stack(scf)
    except HardwareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
