from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flightrl.sixdof import checkpoint_tasks, evaluate_policy, evaluate_teacher, gate_status, load_policy_from_checkpoint
from flightrl.sixdof.tasks import parse_task_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a 6-DoF checkpoint against sim safety gates")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--teacher", action="store_true", help="Evaluate the analytic teacher/reference controller instead of a checkpoint")
    parser.add_argument("--task", default="position_yaw", help="Task spec used with --teacher")
    parser.add_argument("--output", default=None)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--native-step", action="store_true")
    parser.add_argument("--min-clearance-m", type=float, default=0.08)
    parser.add_argument("--min-completed-fraction", type=float, default=0.90)
    parser.add_argument("--max-position-error-m", type=float, default=1.00)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    if args.teacher:
        checkpoint_path = None
        tasks = parse_task_spec(args.task)
        metrics = evaluate_teacher(tasks, seed=args.seed, steps=args.steps, num_envs=args.num_envs, use_native_step=args.native_step)
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required unless --teacher is set")
        checkpoint_path = Path(args.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = load_policy_from_checkpoint(checkpoint)
        tasks = checkpoint_tasks(checkpoint)
        metrics = evaluate_policy(
            model,
            tasks,
            seed=args.seed,
            steps=args.steps,
            num_envs=args.num_envs,
            use_native_step=args.native_step,
        )
    gate = gate_status(
        metrics,
        min_clearance_m=args.min_clearance_m,
        min_completed_fraction=args.min_completed_fraction,
        max_position_error_m=args.max_position_error_m,
    )
    report = {
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "controller": "teacher" if args.teacher else "checkpoint",
        "tasks": list(tasks),
        "steps": args.steps,
        "num_envs": args.num_envs,
        "native_step": args.native_step,
        "thresholds": {
            "min_clearance_m": args.min_clearance_m,
            "min_completed_fraction": args.min_completed_fraction,
            "max_position_error_m": args.max_position_error_m,
        },
        "gate": gate,
        "metrics": metrics,
        "safety": "Simulation gate only; a pass does not approve live hardware deployment.",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        print(f"wrote {output}")
    else:
        print(text)
    if args.fail_on_gate and not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
