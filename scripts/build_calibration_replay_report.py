from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from flightrl.hardware.calibration_quality import summarize_calibration_log
from flightrl.replay import aligned_compare
from flightrl.sixdof.command_replay import load_box_room, replay_velocity_commands, write_csv
from flightrl.sixdof.command_replay_sweep import candidate_grid, sweep_command_replay

from build_sixdof_readiness_report import build_report as build_readiness_report, render_markdown as render_readiness_markdown
from sweep_crazyflie_command_replay import parse_optional_floats, render_markdown as render_sweep_markdown
from summarize_crazyflie_calibration import render_markdown as render_quality_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Build replay evidence from a Crazyflie calibration-flight log")
    parser.add_argument("--input", required=True)
    parser.add_argument("--room-report", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--native-parity", required=True)
    parser.add_argument("--profile-matrix", default=None)
    parser.add_argument("--output-dir", default="artifacts/replay/calibration_pipeline")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--allow-unready-quality", action="store_true")
    parser.add_argument("--override-z-m", type=float, default=None)
    parser.add_argument("--hold-z-values", nargs="+", default=["none"])
    parser.add_argument("--velocity-gains", nargs="+", type=float, default=[0.5, 0.75, 1.25, 2.5])
    parser.add_argument("--yawrate-scales", nargs="+", type=float, default=[1.0, 1.25, 1.5])
    parser.add_argument("--max-dt-values", nargs="+", type=float, default=[0.05])
    parser.add_argument("--command-frames", nargs="+", choices=("body", "world"), default=["body"])
    parser.add_argument("--yaw-sources", nargs="+", choices=("logged", "sim"), default=["logged"])
    parser.add_argument("--vx-signs", nargs="+", type=float, choices=(-1.0, 1.0), default=[1.0])
    parser.add_argument("--vy-signs", nargs="+", type=float, choices=(-1.0, 1.0), default=[1.0])
    args = parser.parse_args()

    outputs = build_pipeline(args)
    print(f"quality={outputs['quality']}")
    print(f"sweep={outputs['sweep']}")
    print(f"comparison={outputs['comparison']}")
    print(f"readiness={outputs['readiness']}")


def build_pipeline(args: argparse.Namespace) -> dict[str, Path]:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    prefix = args.prefix or input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(input_path)
    quality_report = {"input": str(input_path), "summary": summarize_calibration_log(rows)}
    quality_path = output_dir / f"{prefix}.quality.json"
    write_json(quality_path, quality_report)
    quality_path.with_suffix(".md").write_text(render_quality_markdown(quality_report))
    if not quality_report["summary"]["replay_calibration_ready"] and not args.allow_unready_quality:
        raise SystemExit(f"calibration quality failed: {quality_report['summary']['failures']}")

    room = load_box_room(args.room_report)
    candidates = candidate_grid(
        velocity_gains=args.velocity_gains,
        yawrate_scales=args.yawrate_scales,
        max_dt_values=args.max_dt_values,
        override_z_m=args.override_z_m,
        hold_z_values=parse_optional_floats(args.hold_z_values),
        command_frames=args.command_frames,
        yaw_sources=args.yaw_sources,
        vx_signs=args.vx_signs,
        vy_signs=args.vy_signs,
    )
    records = sweep_command_replay(rows, room=room, candidates=candidates)
    sweep_report = {"input": str(input_path), "room_report": args.room_report, "records": records, "best": records[0] if records else None}
    sweep_path = output_dir / f"{prefix}.sweep.json"
    write_json(sweep_path, sweep_report)
    sweep_path.with_suffix(".md").write_text(render_sweep_markdown(sweep_report))

    best = sweep_report["best"]["params"]
    sim_rows, real_rows = replay_velocity_commands(rows, room=room, **best)
    sim_path = output_dir / f"{prefix}.best_sim.csv"
    real_path = output_dir / f"{prefix}.best_real.csv"
    write_csv(sim_path, sim_rows)
    write_csv(real_path, real_rows)
    comparison = {"real": {}, "sim": {}, "aligned": aligned_compare(real_rows, sim_rows)}
    comparison_path = output_dir / f"{prefix}.comparison.json"
    write_json(comparison_path, comparison)

    readiness = build_readiness_report(readiness_args(args, comparison_path))
    readiness_path = output_dir / f"{prefix}.readiness.json"
    write_json(readiness_path, readiness)
    readiness_path.with_suffix(".md").write_text(render_readiness_markdown(readiness) + "\n")
    return {"quality": quality_path, "sweep": sweep_path, "comparison": comparison_path, "readiness": readiness_path}


def readiness_args(args: argparse.Namespace, comparison_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        matrix=args.matrix,
        room_report=args.room_report,
        native_parity=args.native_parity,
        profile_matrix=args.profile_matrix,
        replay_comparison=str(comparison_path),
        residual_sweep=None,
        training_throughput=None,
        puffer_export=None,
        require_replay_comparison=True,
        require_training_throughput=False,
        require_puffer_export=False,
        max_latency_us=50.0,
        max_native_state_rmse=1e-5,
        max_native_range_rmse=1.0,
        max_replay_state_rmse=0.5,
        max_replay_range_rmse_mm=300.0,
        min_replay_overlap_s=1.0,
        min_training_total_sps=0.0,
    )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
