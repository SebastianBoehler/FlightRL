from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.puffer4_door_selection import (
    build_fixed_door_selection_report,
    write_exclusive_selection_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a fixed-door checkpoint from held-out evidence"
    )
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--screen-seed11-report", type=Path, required=True)
    parser.add_argument("--screen-seed23-report", type=Path, required=True)
    parser.add_argument("--screen-seed47-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screens = {
        11: args.screen_seed11_report,
        23: args.screen_seed23_report,
        47: args.screen_seed47_report,
    }
    report = build_fixed_door_selection_report(
        candidate_checkpoint=args.candidate_checkpoint,
        candidate_report=args.candidate_report,
        baseline_checkpoint=args.baseline_checkpoint,
        baseline_report=args.baseline_report,
        screens=screens,
    )
    write_exclusive_selection_report(
        args.output,
        report,
        input_paths=(
            args.candidate_checkpoint,
            args.candidate_report,
            args.baseline_checkpoint,
            args.baseline_report,
            *screens.values(),
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
