from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.sixdof.puffer_calibration import PhysicsSweepGrid, calibrate_puffer_physics, candidate_profiles, render_calibration_markdown
from flightrl.sixdof.puffer_evaluation import PufferEvalConfig
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy
from flightrl.sixdof.tasks import parse_task_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Python six-DoF physics parameters against a MuJoCo Puffer checkpoint gate.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-output")
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--reset-profile", default="obstacle_hover_live")
    parser.add_argument("--sensor-profile", default=None)
    parser.add_argument("--base-physics-profile", default="baseline")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--linear-drag", type=float, nargs="+", default=[0.04, 0.06, 0.08, 0.10])
    parser.add_argument("--rate-tau-s", type=float, nargs="+", default=[0.035, 0.045, 0.060])
    parser.add_argument("--motor-tau-s", type=float, nargs="+", default=[0.0, 0.035, 0.060])
    parser.add_argument("--thrust-scale", type=float, nargs="+", default=[0.75])
    parser.add_argument("--max-open-space-horizontal-speed-p95-m-s", type=float, default=0.75)
    args = parser.parse_args()

    policy = load_puffer_sixdof_policy(args.checkpoint)
    tasks = parse_task_spec(args.task)
    if policy.checkpoint_metadata is None or tasks != policy.checkpoint_metadata.tasks:
        raise SystemExit("--task must exactly match the task contract stored in the Puffer checkpoint")
    config = PufferEvalConfig(
        task=args.task,
        backend="both",
        steps=args.steps,
        num_envs=args.num_envs,
        seed=args.seed,
        reset_profile=args.reset_profile,
        sensor_profile=args.sensor_profile,
        max_open_space_horizontal_speed_p95_m_s=args.max_open_space_horizontal_speed_p95_m_s,
    )
    grid = PhysicsSweepGrid(
        linear_drag=tuple(args.linear_drag),
        rate_tau_s=tuple(args.rate_tau_s),
        motor_tau_s=tuple(args.motor_tau_s),
        thrust_scale=tuple(args.thrust_scale),
    )
    profiles = candidate_profiles(args.base_physics_profile, grid)
    report = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "task": args.task,
        "reset_profile": args.reset_profile,
        "sensor_profile": args.sensor_profile,
        "base_physics_profile": args.base_physics_profile,
        "grid": {
            "linear_drag": list(grid.linear_drag),
            "rate_tau_s": list(grid.rate_tau_s),
            "motor_tau_s": list(grid.motor_tau_s),
            "thrust_scale": list(grid.thrust_scale),
        },
        **calibrate_puffer_physics(policy, config, profiles),
        "safety": "Offline Python-vs-MuJoCo calibration only; passing this report does not approve live hardware deployment.",
    }
    write_outputs(report, args)


def write_outputs(report: dict, args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_calibration_markdown(report) + "\n")
    if args.profile_output and report.get("best"):
        profile_path = Path(args.profile_output)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps({"physics_profile": report["best"]["physics_profile"]}, indent=2, sort_keys=True) + "\n")
    best_score = report["best"]["score"] if report.get("best") else None
    print(f"physics_sweep={output}")
    print(f"candidates={len(report.get('records', []))} best_score={best_score}")


if __name__ == "__main__":
    main()
