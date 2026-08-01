from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.pipeline_verify import verify_pipeline, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that sim-to-real pipeline input fingerprints are still current")
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = verify_pipeline(args.pipeline)
    output = args.output or args.pipeline.with_suffix(".verify.json")
    write_report(report, output)
    print(f"verification={output}")
    print(f"passed={report['passed']}")
    print(f"failures={','.join(report['failures']) or 'none'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
