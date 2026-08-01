from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.semantic.readiness import (
    build_bounded_forward_readiness,
    write_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind semantic simulation and replay evidence to bounded control"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_bounded_forward_readiness(
        args.checkpoint,
        args.training_report,
        args.replay_report,
    )
    output = write_readiness(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output}")


if __name__ == "__main__":
    main()
