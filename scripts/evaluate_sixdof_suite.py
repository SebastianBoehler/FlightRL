from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flightrl.sixdof import checkpoint_tasks, evaluate_policy, evaluate_teacher, gate_status, load_policy_from_checkpoint
from flightrl.sixdof.tasks import parse_task_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a suite of 6-DoF teachers/checkpoints with one gate contract")
    parser.add_argument("--teacher", nargs=2, action="append", default=[], metavar=("LABEL", "TASKS"))
    parser.add_argument("--candidate", nargs=3, action="append", default=[], metavar=("LABEL", "CHECKPOINT", "TASKS"))
    parser.add_argument("--output", default="artifacts/replay/sixdof_validation_suite.json")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--native-step", action="store_true")
    parser.add_argument("--min-clearance-m", type=float, default=0.08)
    parser.add_argument("--min-completed-fraction", type=float, default=0.90)
    parser.add_argument("--max-position-error-m", type=float, default=1.00)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    if not args.teacher and not args.candidate:
        raise SystemExit("provide at least one --teacher or --candidate")

    thresholds = {
        "min_clearance_m": args.min_clearance_m,
        "min_completed_fraction": args.min_completed_fraction,
        "max_position_error_m": args.max_position_error_m,
    }
    records = []
    for idx, (label, task_spec) in enumerate(args.teacher):
        tasks = parse_task_spec(task_spec)
        metrics = evaluate_teacher(tasks, seed=args.seed + idx, steps=args.steps, num_envs=args.num_envs, use_native_step=args.native_step)
        records.append(build_record(label, "teacher", None, tasks, metrics, thresholds))
    offset = len(records)
    for idx, (label, checkpoint_path, task_spec) in enumerate(args.candidate):
        records.append(evaluate_candidate(label, Path(checkpoint_path), task_spec, args, thresholds, args.seed + offset + idx))

    report = {
        "steps": args.steps,
        "num_envs": args.num_envs,
        "native_step": args.native_step,
        "thresholds": thresholds,
        "records": records,
        "summary": {
            "total": len(records),
            "passed": sum(1 for record in records if record["gate"]["passed"]),
            "failed": sum(1 for record in records if not record["gate"]["passed"]),
        },
        "safety": "Simulation validation only; passing this suite does not approve live hardware deployment.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"summary={output}")
    print(f"markdown={output.with_suffix('.md')}")
    if args.fail_on_gate and report["summary"]["failed"]:
        raise SystemExit(2)


def evaluate_candidate(label: str, checkpoint_path: Path, task_spec: str, args: argparse.Namespace, thresholds: dict, seed: int) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy_tasks = checkpoint_tasks(checkpoint)
    tasks = policy_tasks if task_spec == "checkpoint" else parse_task_spec(task_spec)
    model = load_policy_from_checkpoint(checkpoint)
    metrics = evaluate_policy(
        model,
        policy_tasks,
        seed=seed,
        steps=args.steps,
        num_envs=args.num_envs,
        use_native_step=args.native_step,
        eval_tasks=tasks,
    )
    return build_record(label, "checkpoint", checkpoint_path, tasks, metrics, thresholds)


def build_record(label: str, controller: str, checkpoint: Path | None, tasks: tuple[str, ...], metrics: dict, thresholds: dict) -> dict:
    return {
        "label": label,
        "controller": controller,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "tasks": list(tasks),
        "gate": gate_status(metrics, **thresholds),
        "metrics": metrics,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# 6-DoF Validation Suite",
        "",
        "| label | controller | tasks | passed | failures | pos err m | clearance p01 m | completed | action sat | teacher L2 |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in report["records"]:
        metrics = record["metrics"]
        gate = record["gate"]
        lines.append(
            f"| {record['label']} | {record['controller']} | {', '.join(record['tasks'])} | {gate['passed']} | "
            f"{', '.join(gate['failures']) or 'none'} | {metrics['mean_position_error_m']:.4f} | "
            f"{metrics.get('clearance_p01_m', metrics['min_clearance_m']):.4f} | "
            f"{metrics['mean_completed_fraction']:.4f} | {metrics.get('action_saturation_fraction', 0.0):.4f} | "
            f"{metrics.get('teacher_action_l2_mean', 0.0):.4f} |"
        )
    summary = report["summary"]
    lines.extend(
        [
            "",
            f"Passed `{summary['passed']}` of `{summary['total']}` records.",
            "",
            report["safety"],
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
