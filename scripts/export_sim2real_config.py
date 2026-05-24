from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.profile_export import export_config, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a simulator TOML config from a ready sim-to-real profile")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=Path("configs/tasks/crazyflie_hover.toml"))
    parser.add_argument("--output-config", type=Path, default=Path("configs/hardware/measured_crazyflie_sim.toml"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/replay/sim2real_config_export.json"))
    args = parser.parse_args()

    report = export_config(args.profile, base_config=args.base_config, output_config=args.output_config)
    write_report(report, args.report)
    print(f"report={args.report}")
    print(f"markdown={args.report.with_suffix('.md')}")
    print(f"exported={report['exported']}")


if __name__ == "__main__":
    main()
