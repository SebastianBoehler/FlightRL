from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.hardware.paired_capture_validation import validate_paired_capture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a stationary AI Deck plus telemetry capture."
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    report = validate_paired_capture(args.run_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.run_dir / "validation.json").write_text(rendered)
    print(rendered, end="")
    return 0 if report["paired_capture_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
