from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.checkpoint_manifest import build_checkpoint_manifest, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a checkpoint manifest split by sim-ready and hardware-approved status")
    parser.add_argument("--transfer-gate", type=Path, required=True)
    parser.add_argument("--sim-readiness", type=Path, required=True)
    parser.add_argument("--deployment-readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/replay/sim2real_checkpoint_manifest.json"))
    args = parser.parse_args()

    report = build_checkpoint_manifest(
        transfer_gate=args.transfer_gate,
        sim_readiness=args.sim_readiness,
        deployment_readiness=args.deployment_readiness,
    )
    write_report(report, args.output)
    print(f"manifest={args.output}")
    print(f"markdown={args.output.with_suffix('.md')}")
    print(f"hardware_approved={report['summary']['hardware_approved']}")


if __name__ == "__main__":
    main()
