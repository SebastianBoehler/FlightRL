from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.latency import summarize_latency, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Crazyflie command-to-state latency from a replay CSV")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/replay/hardware_latency_summary.json"))
    parser.add_argument("--max-lag-s", type=float, default=0.5)
    parser.add_argument("--min-abs-corr", type=float, default=0.35)
    parser.add_argument("--max-median-latency-s", type=float, default=0.25)
    args = parser.parse_args()

    report = summarize_latency(
        args.input,
        max_lag_s=args.max_lag_s,
        min_abs_corr=args.min_abs_corr,
        max_median_latency_s=args.max_median_latency_s,
    )
    write_report(report, args.output)
    print(f"summary={args.output}")
    print(f"markdown={args.output.with_suffix('.md')}")
    print(f"latency_ready={report['summary']['latency_ready']}")


if __name__ == "__main__":
    main()
