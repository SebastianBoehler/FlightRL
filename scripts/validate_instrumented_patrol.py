from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.hardware.flight_validation import validate_instrumented_patrol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an instrumented short Crazyflie patrol."
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    report = validate_instrumented_patrol(args.run_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.run_dir / "validation.json").write_text(rendered)
    print(rendered, end="")
    return 0 if report["instrumented_patrol_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
