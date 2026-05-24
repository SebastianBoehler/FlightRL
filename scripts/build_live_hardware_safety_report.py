from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.live_safety import build_live_safety_report, write_report


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit live Crazyflie scripts for learned-checkpoint hardware safety gates")
    parser.add_argument("--script", action="append", type=Path, default=None, help="Script to scan. Defaults to scripts/crazyflie_*.py.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/replay/live_hardware_safety_current.json")
    args = parser.parse_args()

    paths = args.script or sorted((ROOT / "scripts").glob("crazyflie_*.py"))
    report = build_live_safety_report([resolve_path(path) for path in paths])
    write_report(report, args.output)
    print(f"live_safety={args.output}")
    print(f"markdown={args.output.with_suffix('.md')}")
    print(f"passed={report['summary']['passed']}")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
