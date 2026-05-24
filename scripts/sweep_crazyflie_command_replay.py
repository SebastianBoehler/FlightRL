from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.sixdof.command_replay import load_box_room, load_csv, replay_velocity_commands, write_csv
from flightrl.sixdof.command_replay_sweep import candidate_grid, sweep_command_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Crazyflie command replay bridge parameters against real logs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--room-report", default=None)
    parser.add_argument("--output", default="artifacts/replay/crazyflie_command_replay_sweep.json")
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--best-sim-output", default=None)
    parser.add_argument("--best-real-output", default=None)
    parser.add_argument("--override-z-m", type=float, default=None)
    parser.add_argument("--hold-z-values", nargs="+", default=["none"])
    parser.add_argument("--velocity-gains", nargs="+", type=float, default=[1.0, 2.5, 4.0])
    parser.add_argument("--yawrate-scales", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--max-dt-values", nargs="+", type=float, default=[0.05, 0.08])
    parser.add_argument("--command-frames", nargs="+", choices=("body", "world"), default=["body"])
    parser.add_argument("--yaw-sources", nargs="+", choices=("logged", "sim"), default=["logged"])
    parser.add_argument("--vx-signs", nargs="+", type=float, choices=(-1.0, 1.0), default=[1.0])
    parser.add_argument("--vy-signs", nargs="+", type=float, choices=(-1.0, 1.0), default=[1.0])
    args = parser.parse_args()

    rows = load_csv(args.input)
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
    report = {"input": args.input, "room_report": args.room_report, "records": records, "best": records[0] if records else None}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    markdown = Path(args.markdown or output.with_suffix(".md"))
    markdown.write_text(render_markdown(report))
    if records and args.best_sim_output:
        write_best_rollout(rows, room, records[0]["params"], args.best_sim_output, args.best_real_output)
    print(f"wrote {output} and {markdown}; candidates={len(records)}")


def parse_optional_floats(values: list[str]) -> list[float | None]:
    return [None if value.lower() in {"none", "null"} else float(value) for value in values]


def write_best_rollout(rows, room, params: dict, sim_output: str, real_output: str | None) -> None:
    sim_rows, real_rows = replay_velocity_commands(rows, room=room, **params)
    write_csv(sim_output, sim_rows)
    if real_output:
        write_csv(real_output, real_rows)


def render_markdown(report: dict) -> str:
    lines = [
        "# Crazyflie Command Replay Sweep",
        "",
        "| rank | score | velocity_gain | yawrate_scale | max_dt_s | hold_z_m | frame/yaw | vx/vy sign | xy_rmse_m | z_rmse_m | yaw_rmse_deg | range_rmse_mm |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, record in enumerate(report["records"][:12], start=1):
        params = record["params"]
        metrics = record["metrics"]
        lines.append(
            f"| {rank} | {record['score']:.4f} | {params['velocity_gain']:.3f} | {params['yawrate_scale']:.3f} | "
            f"{params['max_dt_s']:.3f} | {params['hold_z_m']} | {params.get('command_frame', 'body')}/{params.get('yaw_source', 'logged')} | "
            f"{params.get('vx_sign', 1.0):.0f}/{params.get('vy_sign', 1.0):.0f} | {metrics['worst_xy_state_rmse_m']:.4f} | "
            f"{metrics['z_rmse_m']:.4f} | {metrics['yaw_rmse_deg']:.2f} | {metrics['worst_range_rmse_mm']:.1f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
