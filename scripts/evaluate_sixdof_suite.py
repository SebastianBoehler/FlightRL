from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flightrl.sixdof import checkpoint_tasks, evaluate_checkpoint_policy, evaluate_teacher, gate_status
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
    parser.add_argument("--reset-profile", default=None)
    parser.add_argument("--min-clearance-m", type=float, default=0.08)
    parser.add_argument("--min-completed-fraction", type=float, default=0.90)
    parser.add_argument("--max-position-error-m", type=float, default=1.00)
    parser.add_argument("--max-yaw-error-rad", type=float, default=None)
    parser.add_argument("--max-yaw-p95-error-rad", type=float, default=None)
    parser.add_argument("--max-settled-yaw-p95-error-rad", type=float, default=None)
    parser.add_argument("--metric-start-step", type=int, default=0)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    if not args.teacher and not args.candidate:
        raise SystemExit("provide at least one --teacher or --candidate")

    thresholds = {
        "min_clearance_m": args.min_clearance_m,
        "min_completed_fraction": args.min_completed_fraction,
        "max_position_error_m": args.max_position_error_m,
        "max_yaw_error_rad": args.max_yaw_error_rad,
        "max_yaw_p95_error_rad": args.max_yaw_p95_error_rad,
        "max_settled_yaw_p95_error_rad": args.max_settled_yaw_p95_error_rad,
    }
    records = []
    for idx, (label, task_spec) in enumerate(args.teacher):
        tasks = parse_task_spec(task_spec)
        metrics = evaluate_teacher(tasks, seed=args.seed + idx, steps=args.steps, num_envs=args.num_envs, use_native_step=args.native_step, reset_profile=args.reset_profile, metric_start_step=args.metric_start_step)
        records.append(build_record(label, "teacher", None, tasks, metrics, thresholds))
    offset = len(records)
    for idx, (label, checkpoint_path, task_spec) in enumerate(args.candidate):
        records.append(evaluate_candidate(label, Path(checkpoint_path), task_spec, args, thresholds, args.seed + offset + idx))

    report = {
        "steps": args.steps,
        "num_envs": args.num_envs,
        "native_step": args.native_step,
        "reset_profile": args.reset_profile or "broad",
        "thresholds": thresholds,
        "records": records,
        "summary": {
            "total": len(records),
            "passed": sum(1 for record in records if record["gate"]["passed"]),
            "failed": sum(1 for record in records if not record["gate"]["passed"]),
            "best_checkpoint_by_task": best_checkpoint_by_task(records),
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
    metrics = evaluate_checkpoint_policy(
        checkpoint,
        seed=seed,
        steps=args.steps,
        num_envs=args.num_envs,
        use_native_step=args.native_step,
        eval_tasks=tasks,
        reset_profile=args.reset_profile,
        metric_start_step=args.metric_start_step,
    )
    return build_record(label, checkpoint.get("controller", "checkpoint"), checkpoint_path, tasks, metrics, thresholds)


def build_record(label: str, controller: str, checkpoint: Path | None, tasks: tuple[str, ...], metrics: dict, thresholds: dict) -> dict:
    return {
        "label": label,
        "controller": controller,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "tasks": list(tasks),
        "gate": gate_status(metrics, **thresholds),
        "per_task_gate": per_task_gate(metrics, thresholds),
        "metrics": metrics,
    }


def per_task_gate(metrics: dict, thresholds: dict) -> dict[str, dict]:
    return {task: gate_status(normalize_task_metrics(values), **thresholds) for task, values in metrics.get("per_task", {}).items()}


def normalize_task_metrics(metrics: dict) -> dict:
    return {
        **metrics,
        "mean_completed_fraction": metrics["completed_fraction"],
        "mean_survival_fraction": metrics.get("survival_fraction", metrics["completed_fraction"]),
    }


def best_checkpoint_by_task(records: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for record in records:
        if record["controller"] == "teacher" or len(record["tasks"]) != 1:
            continue
        task = record["tasks"][0]
        candidate = compact_candidate(record)
        if task not in best or candidate_score(candidate) < candidate_score(best[task]):
            best[task] = candidate
    return best


def compact_candidate(record: dict) -> dict:
    metrics = record["metrics"]
    return {
        "label": record["label"],
        "checkpoint": record["checkpoint"],
        "passed": record["gate"]["passed"],
        "failures": record["gate"]["failures"],
        "mean_position_error_m": metrics["mean_position_error_m"],
        "mean_yaw_error_rad": metrics.get("mean_yaw_error_rad"),
        "yaw_error_p95_rad": metrics.get("yaw_error_p95_rad"),
        "clearance_p01_m": metrics.get("clearance_p01_m", metrics["min_clearance_m"]),
        "mean_completed_fraction": metrics["mean_completed_fraction"],
        "mean_survival_fraction": metrics.get("mean_survival_fraction", metrics["mean_completed_fraction"]),
        "teacher_action_l2_mean": metrics.get("teacher_action_l2_mean"),
        "settled_yaw_error_p95_rad": metrics.get("settled_yaw_error_p95_rad"),
    }


def candidate_score(candidate: dict) -> tuple:
    return (
        0 if candidate["passed"] else 1,
        -candidate["mean_survival_fraction"],
        candidate["mean_position_error_m"],
        -candidate["mean_completed_fraction"],
        -candidate["clearance_p01_m"],
    )


def render_markdown(report: dict) -> str:
    lines = [
        "# 6-DoF Validation Suite",
        "",
        "| label | controller | tasks | passed | failures | pos err m | yaw err rad | yaw p95 rad | settled yaw p95 | clearance p01 m | completed | survival | action sat | teacher L2 |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in report["records"]:
        metrics = record["metrics"]
        gate = record["gate"]
        lines.append(
            f"| {record['label']} | {record['controller']} | {', '.join(record['tasks'])} | {gate['passed']} | "
            f"{', '.join(gate['failures']) or 'none'} | {metrics['mean_position_error_m']:.4f} | "
            f"{metrics.get('mean_yaw_error_rad', 0.0):.4f} | "
            f"{metrics.get('yaw_error_p95_rad', 0.0):.4f} | "
            f"{metrics.get('settled_yaw_error_p95_rad', 0.0):.4f} | "
            f"{metrics.get('clearance_p01_m', metrics['min_clearance_m']):.4f} | "
            f"{metrics['mean_completed_fraction']:.4f} | "
            f"{metrics.get('mean_survival_fraction', metrics['mean_completed_fraction']):.4f} | "
            f"{metrics.get('action_saturation_fraction', 0.0):.4f} | "
            f"{metrics.get('teacher_action_l2_mean', 0.0):.4f} |"
        )
    summary = report["summary"]
    lines.extend(
        [
            "",
            f"Passed `{summary['passed']}` of `{summary['total']}` records.",
        ]
    )
    if summary["best_checkpoint_by_task"]:
        lines.extend(["", "## Best Checkpoints By Task", ""])
        for task, candidate in summary["best_checkpoint_by_task"].items():
            lines.append(
                f"- `{task}`: `{candidate['label']}` passed=`{candidate['passed']}` "
                f"pos_err=`{candidate['mean_position_error_m']:.4f}` completed=`{candidate['mean_completed_fraction']:.4f}`"
            )
    if any(record.get("per_task_gate") for record in report["records"]):
        lines.extend(["", "## Per-Task Gates", ""])
        for record in report["records"]:
            if record.get("per_task_gate"):
                lines.append(f"- `{record['label']}`: {format_task_gates(record['per_task_gate'])}")
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def format_task_gates(per_task: dict[str, dict]) -> str:
    parts = []
    for task, gate in per_task.items():
        failures = ",".join(gate["failures"]) or "pass"
        parts.append(f"{task}={failures}")
    return "; ".join(parts)


if __name__ == "__main__":
    main()
