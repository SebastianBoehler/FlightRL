from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.sim2real.blockers import load_hardware_blockers
from flightrl.sim2real.evidence_gap import build_evidence_gap_report, write_report as write_gap_report
from flightrl.sim2real.pipeline import build_pipeline, output_paths


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the offline Crazyflie sim-to-real evidence chain")
    parser.add_argument("--label", default="current_2026-05-20")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/replay")
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "configs/hardware/manufacturer_placeholder.toml")
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/tasks/crazyflie_hover.toml")
    parser.add_argument("--output-config", type=Path, default=ROOT / "configs/hardware/measured_crazyflie_sim.toml")
    parser.add_argument("--motor-calibration", type=Path, default=ROOT / "artifacts/replay/motor_bench_single_motor_calibration_2026-05-20.json")
    parser.add_argument("--stationary-noise", type=Path, default=ROOT / "artifacts/replay/handheld_room_map_sync_noise_rejected_2026-05-20.json")
    parser.add_argument("--hardware-latency", type=Path, default=ROOT / "artifacts/replay/room_scan_autonomous_35s.clean20.hardware_latency.json")
    parser.add_argument("--calibration-quality", type=Path, default=ROOT / "artifacts/replay/room_scan_autonomous_35s.clean20.calibration_quality.json")
    parser.add_argument("--deployment-readiness", type=Path, default=ROOT / "artifacts/replay/sixdof_deployment_readiness_puffer_replay_blocked_2026-05-20.json")
    parser.add_argument("--replay-comparison", type=Path, default=ROOT / "artifacts/replay/room_scan_autonomous_35s.command_replay_frame_best_compare.json")
    parser.add_argument("--motor-bench", type=Path, default=ROOT / "artifacts/crazyflie_logs/motor_bench_single_motor.csv")
    parser.add_argument("--sim-readiness", type=Path, default=ROOT / "artifacts/replay/sixdof_readiness_multitask_residual_puffer_gated_2026-05-20.json")
    parser.add_argument("--room-report", type=Path, default=ROOT / "artifacts/replay/room_scan_autonomous_35s.clean20.strict_path.room.json")
    parser.add_argument("--live-script", action="append", type=Path, default=None)
    parser.add_argument("--hardware-blockers-file", type=Path, default=ROOT / "configs/hardware/current_blockers.json")
    parser.add_argument("--hardware-blocker", action="append", default=[])
    args = parser.parse_args()

    live_scripts = [resolve_path(path) for path in (args.live_script or sorted((ROOT / "scripts").glob("crazyflie_*.py")))]
    hardware_blockers_file = resolve_optional(args.hardware_blockers_file)
    hardware_blockers = load_hardware_blockers(hardware_blockers_file, args.hardware_blocker)
    outputs = output_paths(resolve_path(args.output_dir), args.label)
    report = build_pipeline(
        outputs=outputs,
        hardware_config=resolve_path(args.hardware_config),
        base_config=resolve_path(args.base_config),
        output_config=resolve_path(args.output_config),
        deployment_readiness=resolve_path(args.deployment_readiness),
        sim_readiness=resolve_path(args.sim_readiness),
        live_scripts=live_scripts,
        motor_calibration=resolve_optional(args.motor_calibration),
        stationary_noise=resolve_optional(args.stationary_noise),
        hardware_latency=resolve_optional(args.hardware_latency),
        calibration_quality=resolve_optional(args.calibration_quality),
        replay_comparison=resolve_optional(args.replay_comparison),
        motor_bench=resolve_optional(args.motor_bench),
        room_report=resolve_optional(args.room_report),
        hardware_blockers=hardware_blockers,
        input_paths={
            "hardware_config": resolve_path(args.hardware_config),
            "base_config": resolve_path(args.base_config),
            "output_config": resolve_path(args.output_config),
            "motor_calibration": resolve_optional(args.motor_calibration),
            "stationary_noise": resolve_optional(args.stationary_noise),
            "hardware_latency": resolve_optional(args.hardware_latency),
            "calibration_quality": resolve_optional(args.calibration_quality),
            "deployment_readiness": resolve_path(args.deployment_readiness),
            "replay_comparison": resolve_optional(args.replay_comparison),
            "motor_bench": resolve_optional(args.motor_bench),
            "sim_readiness": resolve_path(args.sim_readiness),
            "room_report": resolve_optional(args.room_report),
            "live_scripts": live_scripts,
            "hardware_blockers_file": hardware_blockers_file,
            "hardware_blockers": hardware_blockers,
        },
    )
    gap_report = build_evidence_gap_report(outputs["pipeline"])
    write_gap_report(gap_report, outputs["evidence_gap"])
    print(f"pipeline={outputs['pipeline']}")
    print(f"gap_report={outputs['evidence_gap']}")
    print(f"transfer_approved={report['transfer_approved']}")
    print(f"hardware_approved={report['hardware_approved_checkpoints']}")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def resolve_optional(path: Path | None) -> Path | None:
    return resolve_path(path) if path else None


if __name__ == "__main__":
    main()
