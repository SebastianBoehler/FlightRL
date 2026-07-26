from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.sixdof.action_calibration import write_scaled_action_head_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale a Puffer checkpoint action head for offline calibration.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--report")
    args = parser.parse_args()

    write_scaled_action_head_checkpoint(args.checkpoint, args.output, args.scale)
    report = {
        "checkpoint": args.checkpoint,
        "output": args.output,
        "scale": args.scale,
        "safety": "Offline checkpoint calibration only; passing downstream gates does not approve live hardware deployment.",
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"scaled_checkpoint={args.output}")
    print(f"scale={args.scale}")


if __name__ == "__main__":
    main()
