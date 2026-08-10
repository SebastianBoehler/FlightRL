from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.exploration.range_shadow import replay_range_shadow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay range exploration policy without controlling a drone"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = replay_range_shadow(args.checkpoint, args.telemetry, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["replay_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
