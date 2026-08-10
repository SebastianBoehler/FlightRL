from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.exploration.range_checkpoint import load_range_checkpoint
from flightrl.exploration.range_evaluation import evaluate_range_candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a range-frontier exploration v2 candidate"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=1_200)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args(argv)
    if args.horizon <= 0 or any(seed < 0 for seed in args.seeds):
        parser.error("evaluation horizon must be positive and seeds nonnegative")
    if args.output.exists():
        parser.error("evaluation output already exists")
    model, _prior = load_range_checkpoint(args.checkpoint)
    report = evaluate_range_candidate(
        model,
        seeds=tuple(args.seeds),
        horizon=args.horizon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"output={args.output}")
    print(f"simulation_gate_passed={report['simulation_gate_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
