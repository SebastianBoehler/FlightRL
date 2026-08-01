from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sixdof.puffer_drone_reference import build_reference_report, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare official Puffer drone against FlightRL 6-DoF env contracts.")
    parser.add_argument("--pufferlib-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/replay/puffer_drone_reference_alignment.json"),
    )
    args = parser.parse_args()

    report = build_reference_report(args.pufferlib_root)
    write_report(report, args.output)
    print(f"puffer_drone_reference_report={args.output}")
    print(f"adaptation_required={report['compatibility']['adaptation_required_for_replacement']}")


if __name__ == "__main__":
    main()
