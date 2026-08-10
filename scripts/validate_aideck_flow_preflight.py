from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.hardware.flow_preflight_validation import validate_flow_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a short props-off AI Deck/Flow raw-motion preflight."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = validate_flow_preflight(args.input)
    output = args.output or args.input.with_suffix(".validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["flow_preflight_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
