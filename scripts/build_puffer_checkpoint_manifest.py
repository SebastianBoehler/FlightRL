from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.puffer_checkpoint_manifest import build_puffer_checkpoint_manifest, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Puffer checkpoint manifest from bundle transfer evidence")
    parser.add_argument("--transfer-gate", type=Path, required=True)
    parser.add_argument("--bundle-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/replay/puffer_checkpoint_manifest.json"))
    args = parser.parse_args()

    report = build_puffer_checkpoint_manifest(transfer_gate=args.transfer_gate, bundle_report=args.bundle_report)
    write_report(report, args.output)
    print(f"manifest={args.output}")
    print(f"markdown={args.output.with_suffix('.md')}")
    print(f"hardware_approved={report['summary']['hardware_approved']}")


if __name__ == "__main__":
    main()
