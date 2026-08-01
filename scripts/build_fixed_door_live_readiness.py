from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.puffer4_door_readiness import (
    bind_fixed_door_readiness_identity,
    build_fixed_door_yaw_readiness,
)
from flightrl.semantic.readiness import write_readiness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind fixed-door simulation and real shadow evidence"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--simulation-report", type=Path, required=True)
    parser.add_argument("--shadow-summary", type=Path, required=True)
    parser.add_argument("--shadow-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_fixed_door_yaw_readiness(
        args.checkpoint,
        args.simulation_report,
        args.shadow_summary,
        args.shadow_csv,
    )
    report = bind_fixed_door_readiness_identity(report, args.output)
    output = write_readiness(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output}")


if __name__ == "__main__":
    main()
