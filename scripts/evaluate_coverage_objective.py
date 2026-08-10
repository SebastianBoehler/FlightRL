from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.exploration.objective_audit import audit_coverage_objective


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sanity-check the privileged coverage objective on procedural rooms."
    )
    parser.add_argument("--seed-start", type=int, default=410)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evidence/coverage_objective_offline.json"),
    )
    args = parser.parse_args(argv)
    if args.seed_start < 0 or args.episodes <= 0:
        parser.error("--seed-start must be non-negative and --episodes must be positive")

    seeds = tuple(range(args.seed_start, args.seed_start + args.episodes))
    report = audit_coverage_objective(seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["objective_sanity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
