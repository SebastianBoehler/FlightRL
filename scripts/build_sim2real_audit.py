from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.sim2real import build_audit, render_markdown
from flightrl.sim2real.blockers import load_hardware_blockers


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sim-to-real evidence audit for Crazyflie transfer readiness")
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "configs/hardware/manufacturer_placeholder.toml")
    parser.add_argument("--calibration-quality", type=Path, default=None)
    parser.add_argument("--deployment-readiness", type=Path, default=None)
    parser.add_argument("--replay-comparison", type=Path, default=None)
    parser.add_argument("--motor-bench", type=Path, default=None)
    parser.add_argument("--stationary-noise", type=Path, default=None)
    parser.add_argument("--hardware-latency", type=Path, default=None)
    parser.add_argument("--sensor-profile", type=Path, default=None)
    parser.add_argument("--hardware-blockers-file", type=Path, default=None)
    parser.add_argument("--hardware-blocker", action="append", default=[])
    parser.add_argument("--max-replay-state-rmse", type=float, default=0.5)
    parser.add_argument("--max-replay-range-rmse-mm", type=float, default=300.0)
    parser.add_argument("--min-motor-powers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/replay/sim2real_audit_current.json")
    args = parser.parse_args()

    report = build_audit(
        hardware_config=args.hardware_config,
        calibration_quality=args.calibration_quality,
        deployment_readiness=args.deployment_readiness,
        replay_comparison=args.replay_comparison,
        motor_bench=args.motor_bench,
        stationary_noise=args.stationary_noise,
        hardware_latency=args.hardware_latency,
        sensor_profile=args.sensor_profile,
        hardware_blockers=load_hardware_blockers(args.hardware_blockers_file, args.hardware_blocker),
        max_replay_state_rmse=args.max_replay_state_rmse,
        max_replay_range_rmse_mm=args.max_replay_range_rmse_mm,
        min_motor_powers=args.min_motor_powers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"audit={args.output}")
    print(f"markdown={args.output.with_suffix('.md')}")
    print(f"transfer_ready={report['transfer_ready']}")
    print(f"blocking_items={','.join(report['blocking_items']) or 'none'}")


if __name__ == "__main__":
    main()
