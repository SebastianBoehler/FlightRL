from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.pipeline_verify import verify_pipeline, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that sim-to-real pipeline input fingerprints are still current")
    parser.add_argument("--pipeline", type=Path, default=Path("artifacts/replay/sim2real_pipeline_current_2026-05-20.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/replay/sim2real_pipeline_current_2026-05-20.verify.json"))
    args = parser.parse_args()

    report = verify_pipeline(args.pipeline)
    write_report(report, args.output)
    print(f"verification={args.output}")
    print(f"passed={report['passed']}")
    print(f"failures={','.join(report['failures']) or 'none'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
