from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.navigation.bundles import build_candidate_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reproducible navigation candidate bundle manifest")
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", default="artifacts/candidates")
    args = parser.parse_args()

    bundle = build_candidate_bundle(
        name=args.name,
        checkpoint=Path(args.checkpoint),
        benchmark_report=Path(args.benchmark),
        output_dir=Path(args.output_dir),
    )
    print(f"candidate_bundle={bundle['files']['manifest']}")
    print(f"markdown={bundle['files']['markdown']}")
    print(f"hardware_eligibility={bundle['hardware_eligibility']}")


if __name__ == "__main__":
    main()
