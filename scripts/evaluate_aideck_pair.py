from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_aideck_grounding import evaluate_paired_captures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate offline separability of operator-labeled AI Deck scenes."
    )
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.sample_count < 4 or args.sample_count % 2:
        parser.error("--sample-count must be an even integer of at least four")

    report = evaluate_paired_captures(
        args.positive,
        args.negative,
        sample_count=args.sample_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
