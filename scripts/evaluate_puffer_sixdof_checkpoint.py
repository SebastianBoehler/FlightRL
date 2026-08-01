from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.sixdof.checkpoint_contract import CHECKPOINT_CONTRACT_ID
from flightrl.sixdof.puffer_evaluation import PufferEvalConfig, evaluate_puffer_backends
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy
from flightrl.sixdof.tasks import parse_task_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PufferLib six-DoF checkpoint in Python and MuJoCo gates.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--backend", choices=("python", "mujoco", "both"), default="both")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--reset-profile", default="obstacle_close_live")
    parser.add_argument("--sensor-profile", default=None)
    parser.add_argument("--physics-profile", default=None)
    parser.add_argument("--domain-randomization", default=None)
    parser.add_argument("--disturbance-profile", default=None)
    parser.add_argument("--min-clearance-m", type=float, default=0.08)
    parser.add_argument("--min-completed-fraction", type=float, default=0.90)
    parser.add_argument("--max-position-error-m", type=float, default=1.00)
    parser.add_argument("--max-horizontal-speed-p95-m-s", type=float, default=1.50)
    parser.add_argument("--max-open-space-horizontal-speed-p95-m-s", type=float)
    parser.add_argument("--max-tilt-p95-deg", type=float, default=35.0)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    tasks = parse_task_spec(args.task)
    if len(tasks) != 1:
        raise SystemExit("Puffer checkpoint gate currently supports one task per run; repeat the command for multi-task checks.")
    policy = load_puffer_sixdof_policy(args.checkpoint)
    if policy.checkpoint_metadata is None or tasks != policy.checkpoint_metadata.tasks:
        raise SystemExit("--task must exactly match the task contract stored in the Puffer checkpoint")
    config = PufferEvalConfig(
        task=tasks[0],
        backend=args.backend,
        steps=args.steps,
        num_envs=args.num_envs,
        seed=args.seed,
        reset_profile=args.reset_profile,
        sensor_profile=args.sensor_profile,
        physics_profile=args.physics_profile,
        domain_randomization=args.domain_randomization,
        disturbance_profile=args.disturbance_profile,
        min_clearance_m=args.min_clearance_m,
        min_completed_fraction=args.min_completed_fraction,
        max_position_error_m=args.max_position_error_m,
        max_horizontal_speed_p95_m_s=args.max_horizontal_speed_p95_m_s,
        max_open_space_horizontal_speed_p95_m_s=args.max_open_space_horizontal_speed_p95_m_s,
        max_tilt_p95_deg=args.max_tilt_p95_deg,
    )
    report = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_contract": CHECKPOINT_CONTRACT_ID,
        "policy": {
            "type": "pufferlib_mlp_mean",
            "observation_dim": policy.metadata.observation_dim,
            "hidden_size": policy.metadata.hidden_size,
            "action_dim": policy.metadata.action_dim,
            "num_layers": policy.metadata.num_layers,
        },
        "tasks": list(tasks),
        "steps": args.steps,
        "num_envs": args.num_envs,
        "reset_profile": args.reset_profile,
        "sensor_profile": args.sensor_profile,
        "physics_profile": args.physics_profile,
        "domain_randomization": args.domain_randomization,
        "disturbance_profile": args.disturbance_profile,
        "thresholds": thresholds(config),
        "reports": evaluate_puffer_backends(policy, config),
        "safety": "Offline simulation gate only; passing this report does not approve live hardware deployment.",
    }
    report["passed"] = all(item.get("gate", {}).get("passed", False) for item in report["reports"].values())
    write_report(report, Path(args.output))
    if args.fail_on_gate and not report["passed"]:
        raise SystemExit(2)


def thresholds(config: PufferEvalConfig) -> dict:
    return {
        "min_clearance_m": config.min_clearance_m,
        "min_completed_fraction": config.min_completed_fraction,
        "max_position_error_m": config.max_position_error_m,
        "max_horizontal_speed_p95_m_s": config.max_horizontal_speed_p95_m_s,
        "max_open_space_horizontal_speed_p95_m_s": config.max_open_space_horizontal_speed_p95_m_s,
        "max_tilt_p95_deg": config.max_tilt_p95_deg,
    }


def write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"puffer_eval={output}")
    print(f"passed={report['passed']}")


def render_markdown(report: dict) -> str:
    lines = ["# Puffer Six-DoF Evaluation", "", f"Passed: `{report['passed']}`", ""]
    for name, item in report["reports"].items():
        if item["status"] != "ok":
            lines.extend([f"## {name}", "", f"Status: `{item['status']}`", ""])
            continue
        metrics = item["metrics"]
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Gate: `{item['gate']['passed']}`",
                f"- Failures: `{', '.join(item['gate']['failures']) or 'none'}`",
                f"- Reward: `{metrics['mean_reward']:.3f}`",
                f"- Position error: `{metrics['mean_position_error_m']:.3f}` m",
                f"- Clearance p01: `{metrics['clearance_p01_m']:.3f}` m",
                f"- Survival: `{metrics['mean_survival_fraction']:.3f}`",
                f"- Horizontal speed p95: `{metrics.get('horizontal_speed_p95_m_s', 0.0):.3f}` m/s",
                f"- Open-space horizontal speed p95: `{metrics.get('open_space_horizontal_speed_p95_m_s', 0.0):.3f}` m/s",
                f"- Tilt p95: `{metrics.get('tilt_p95_deg', 0.0):.1f}` deg",
                "",
            ]
        )
    lines.append(report["safety"])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
