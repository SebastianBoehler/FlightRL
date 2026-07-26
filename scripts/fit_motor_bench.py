from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.actuator import fit_motor_calibration, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Crazyflie prop-off motor RPM curves from motor bench CSV")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/replay/motor_bench_calibration.json"))
    parser.add_argument("--min-powers", type=int, default=3)
    parser.add_argument("--min-r2", type=float, default=0.9)
    parser.add_argument("--max-gain-imbalance", type=float, default=0.25)
    parser.add_argument("--min-valid-rpm", type=float, default=0.0)
    parser.add_argument("--max-dropout-ratio", type=float, default=0.0)
    args = parser.parse_args()

    report = fit_motor_calibration(
        args.input,
        min_powers=args.min_powers,
        min_r2=args.min_r2,
        max_gain_imbalance=args.max_gain_imbalance,
        min_valid_rpm=args.min_valid_rpm,
        max_dropout_ratio=args.max_dropout_ratio,
    )
    write_report(report, args.output)
    print(f"summary={args.output}")
    print(f"markdown={args.output.with_suffix('.md')}")
    print(f"motor_calibration_passed={report['summary']['passed']}")


if __name__ == "__main__":
    main()
