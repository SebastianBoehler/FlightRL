from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.measured_config import build_measured_hardware_config, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose a measured Crazyflie hardware config from offline evidence.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--physics-calibration", type=Path, default=None)
    parser.add_argument("--motor-calibration", type=Path, default=None)
    parser.add_argument("--live-system-id", type=Path, default=None)
    parser.add_argument("--sensor-profile", type=Path, default=None)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = build_measured_hardware_config(
        base_config=args.base_config,
        output_config=args.output_config,
        physics_calibration=args.physics_calibration,
        motor_calibration=args.motor_calibration,
        live_system_id=args.live_system_id,
        sensor_profile=args.sensor_profile,
    )
    write_report(report, args.report)
    print(f"config={args.output_config}")
    print(f"report={args.report}")
    print(f"confidence={report['sim2real']['confidence']}")


if __name__ == "__main__":
    main()
