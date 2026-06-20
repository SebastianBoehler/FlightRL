from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.navigation.benchmark import build_navigation_benchmark_report, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single-drone navigation benchmark report from scenario metrics")
    parser.add_argument("--input", required=True, help="JSON file with a records array")
    parser.add_argument("--output", default="artifacts/replay/navigation_benchmark.json")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    records = payload.get("records")
    if not isinstance(records, list):
        raise SystemExit("--input must contain a records array")
    output = Path(args.output)
    report = build_navigation_benchmark_report(records)
    write_report(report, output)
    print(f"navigation_benchmark={output}")
    print(f"markdown={output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
