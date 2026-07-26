from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.noise import summarize_stationary_noise_logs, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize stationary Crazyflie telemetry noise for simulator randomization")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/replay/stationary_noise_summary.json"))
    parser.add_argument("--min-duration-s", type=float, default=30.0)
    parser.add_argument("--max-position-span-m", type=float, default=0.08)
    parser.add_argument("--max-attitude-span-deg", type=float, default=6.0)
    parser.add_argument("--max-range-span-mm", type=float, default=300.0)
    args = parser.parse_args()

    report = summarize_stationary_noise_logs(
        args.input,
        min_duration_s=args.min_duration_s,
        max_position_span_m=args.max_position_span_m,
        max_attitude_span_deg=args.max_attitude_span_deg,
        max_range_span_mm=args.max_range_span_mm,
    )
    write_report(report, args.output)
    print(f"summary={args.output}")
    print(f"markdown={args.output.with_suffix('.md')}")
    print(f"stationary_noise_ready={report['summary']['stationary_noise_ready']}")


if __name__ == "__main__":
    main()
