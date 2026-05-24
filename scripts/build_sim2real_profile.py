from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.profile import build_profile, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simulator priors from measured sim-to-real evidence")
    parser.add_argument("--hardware-config", type=Path, required=True)
    parser.add_argument("--motor-calibration", type=Path, default=None)
    parser.add_argument("--stationary-noise", type=Path, default=None)
    parser.add_argument("--hardware-latency", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/replay/sim2real_profile.json"))
    args = parser.parse_args()

    report = build_profile(
        hardware_config=args.hardware_config,
        motor_calibration=args.motor_calibration,
        stationary_noise=args.stationary_noise,
        hardware_latency=args.hardware_latency,
    )
    write_report(report, args.output)
    print(f"profile={args.output}")
    print(f"markdown={args.output.with_suffix('.md')}")
    print(f"profile_ready={report['summary']['profile_ready']}")


if __name__ == "__main__":
    main()
