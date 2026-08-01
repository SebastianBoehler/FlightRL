from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.evidence_gap import build_evidence_gap_report, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize remaining evidence gaps before Crazyflie sim-to-real transfer")
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/replay/sim2real_evidence_gap.json"))
    args = parser.parse_args()

    report = build_evidence_gap_report(args.pipeline)
    write_report(report, args.output)
    print(f"gap_report={args.output}")
    print(f"decision={report['decision']}")
    print(f"enough_for_one_step_transfer={report['enough_for_one_step_transfer']}")


if __name__ == "__main__":
    main()
