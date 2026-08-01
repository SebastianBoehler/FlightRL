from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.sim2real.data_plan import build_data_plan, render_markdown


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the next-data plan for closing sim-to-real audit blockers")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--motor-bench", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/replay/sim2real_data_plan_current.json")
    args = parser.parse_args()

    report = build_data_plan(args.audit, motor_bench=args.motor_bench)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"data_plan={args.output}")
    print(f"markdown={args.output.with_suffix('.md')}")
    print(f"next_actions={len(report['next_actions'])}")


if __name__ == "__main__":
    main()
