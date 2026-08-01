from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.semantic.readiness import (
    build_yaw_only_readiness,
    write_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine semantic simulation and replay evidence into a live gate"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sim-report", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_yaw_only_readiness(
        args.checkpoint,
        args.sim_report,
        args.replay_report,
    )
    output = write_readiness(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output}")


if __name__ == "__main__":
    main()
