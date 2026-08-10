from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.exploration.behavior_audit import audit_scan_advance_behavior


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the recognizable scan-advance-turn simulation behavior."
    )
    parser.add_argument("--seed-start", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evidence/scan_advance_behavior_offline.json"),
    )
    args = parser.parse_args(argv)
    if args.seed_start < 0 or args.episodes <= 0 or args.steps <= 0:
        parser.error("seed start must be non-negative; episodes and steps must be positive")

    report = audit_scan_advance_behavior(
        tuple(range(args.seed_start, args.seed_start + args.episodes)),
        maximum_steps=args.steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["recognizable_behavior_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
