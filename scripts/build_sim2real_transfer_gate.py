from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.transfer_gate import build_transfer_gate, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final sim-to-real transfer gate from audit/profile/readiness evidence")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--config-export", type=Path, required=True)
    parser.add_argument("--deployment-readiness", type=Path, required=True)
    parser.add_argument("--sim-readiness", type=Path, default=None)
    parser.add_argument("--room-report", type=Path, default=None)
    parser.add_argument("--live-safety", type=Path, default=None)
    parser.add_argument("--puffer-transfer-test", type=Path, action="append", default=None)
    parser.add_argument("--hardware-blockers", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/replay/sim2real_transfer_gate.json"))
    args = parser.parse_args()

    report = build_transfer_gate(
        audit=args.audit,
        profile=args.profile,
        config_export=args.config_export,
        deployment_readiness=args.deployment_readiness,
        sim_readiness=args.sim_readiness,
        room_report=args.room_report,
        live_safety=args.live_safety,
        puffer_transfer_test=args.puffer_transfer_test,
        hardware_blockers=args.hardware_blockers,
    )
    write_report(report, args.output)
    print(f"gate={args.output}")
    print(f"markdown={args.output.with_suffix('.md')}")
    print(f"transfer_approved={report['transfer_approved']}")


if __name__ == "__main__":
    main()
