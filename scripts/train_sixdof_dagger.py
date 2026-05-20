from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flightrl.sixdof import evaluate_policy, gate_status, load_policy_from_checkpoint
from flightrl.sixdof.dagger import collect_policy_dataset, merge_datasets
from flightrl.sixdof.dataset import load_dataset, parse_task_probabilities, write_dataset
from flightrl.sixdof.evaluation import checkpoint_tasks
from flightrl.sixdof.offline import OfflineTrainConfig, train_offline_policy
from flightrl.sixdof.tasks import parse_task_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Iteratively train 6-DoF policies with DAgger policy-state datasets")
    parser.add_argument("--seed-dataset", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/dagger/sixdof")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--task", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=511)
    parser.add_argument("--eval-steps", type=int, default=800)
    parser.add_argument("--eval-num-envs", type=int, default=128)
    parser.add_argument("--select-by-eval", action="store_true")
    parser.add_argument("--native-step", action="store_true")
    parser.add_argument("--reset-profile", default=None, help="Named reset curriculum for DAgger rollout collection.")
    parser.add_argument("--eval-reset-profile", default=None, help="Named reset profile used for eval-based selection and iteration reports.")
    parser.add_argument("--action-weighting", default="none", choices=("none", "inverse_std"))
    parser.add_argument("--min-clearance-m", type=float, default=0.08)
    parser.add_argument("--min-completed-fraction", type=float, default=0.90)
    parser.add_argument("--max-position-error-m", type=float, default=1.00)
    parser.add_argument("--task-weight", action="append", default=[], metavar="TASK=WEIGHT", help="Per-task sample weight for offline retraining. Repeatable.")
    parser.add_argument("--task-probability", action="append", default=[], metavar="TASK=WEIGHT", help="Relative DAgger rollout sampling weight. Unspecified tasks keep weight 1.0.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    current_checkpoint = Path(args.initial_checkpoint)
    dataset_paths = [Path(args.seed_dataset)]
    reports = []
    task_weights = parse_task_weights(args.task_weight)
    task_probabilities = parse_task_probabilities(args.task_probability)
    for iteration in range(1, args.iterations + 1):
        dagger_dataset = collect_policy_dataset(
            checkpoint_path=current_checkpoint,
            task_spec=args.task,
            num_envs=args.num_envs,
            steps=args.steps,
            seed=args.seed + iteration,
            use_native_step=args.native_step,
            beta=args.beta,
            reset_profile=args.reset_profile,
            task_probabilities=task_probabilities,
        )
        merged = merge_datasets(dataset_paths, dagger_dataset)
        dataset_path = write_dataset(output_dir / f"iter_{iteration:02d}.npz", merged)
        checkpoint_path = output_dir / f"iter_{iteration:02d}.pt"
        train_config = OfflineTrainConfig(
            dataset=str(dataset_path),
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_ratio=args.val_ratio,
            seed=args.seed + 100 * iteration,
            eval_steps=args.eval_steps,
            eval_num_envs=args.eval_num_envs,
            select_by_eval=args.select_by_eval,
            use_native_step=args.native_step,
            eval_reset_profile=args.eval_reset_profile,
            action_weighting=args.action_weighting,
            task_weights=task_weights,
        )
        checkpoint = train_offline_policy(load_dataset(dataset_path), train_config)
        torch.save(checkpoint, checkpoint_path)
        report = evaluate_checkpoint(
            checkpoint,
            checkpoint_path,
            args,
            seed=args.seed + 1000 * iteration,
        )
        report["iteration"] = iteration
        report["dataset"] = str(dataset_path)
        report["checkpoint"] = str(checkpoint_path)
        report["val_loss"] = float(checkpoint["val_loss"])
        report["dagger_metadata"] = dagger_dataset["metadata"]
        reports.append(report)
        (output_dir / f"iter_{iteration:02d}.report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print_status(report)
        current_checkpoint = checkpoint_path
        dataset_paths = [dataset_path]
    summary = {"iterations": reports, "best": select_best(reports)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"summary={output_dir / 'summary.json'}")


def evaluate_checkpoint(checkpoint: dict, checkpoint_path: Path, args, *, seed: int) -> dict:
    model = load_policy_from_checkpoint(checkpoint)
    checkpoint_task_list = checkpoint_tasks(checkpoint)
    tasks = parse_task_spec(args.task) if args.task else checkpoint_task_list
    metrics = evaluate_policy(
        model,
        checkpoint_task_list,
        seed=seed,
        steps=args.eval_steps,
        num_envs=args.eval_num_envs,
        use_native_step=args.native_step,
        eval_tasks=tasks,
        reset_profile=args.eval_reset_profile,
        observation_mode=checkpoint.get("observation_mode", "base"),
    )
    gate = gate_status(
        metrics,
        min_clearance_m=args.min_clearance_m,
        min_completed_fraction=args.min_completed_fraction,
        max_position_error_m=args.max_position_error_m,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "tasks": list(tasks),
        "steps": args.eval_steps,
        "num_envs": args.eval_num_envs,
        "native_step": args.native_step,
        "eval_reset_profile": args.eval_reset_profile or "broad",
        "gate": gate,
        "metrics": metrics,
        "safety": "Simulation gate only; a pass does not approve live hardware deployment.",
    }


def select_best(reports: list[dict]) -> dict | None:
    if not reports:
        return None
    return min(
        reports,
        key=lambda report: (
            0 if report["gate"]["passed"] else 1,
            -report["metrics"].get("mean_survival_fraction", report["metrics"]["mean_completed_fraction"]),
            report["metrics"]["mean_position_error_m"],
            -report["metrics"]["mean_completed_fraction"],
        ),
    )


def print_status(report: dict) -> None:
    metrics = report["metrics"]
    print(
        "iter={iteration} checkpoint={checkpoint} gate={passed} "
        "position_error={position_error:.3f} completed={completed:.3f} clearance_p01={clearance:.3f}".format(
            iteration=report["iteration"],
            checkpoint=report["checkpoint"],
            passed=report["gate"]["passed"],
            position_error=metrics["mean_position_error_m"],
            completed=metrics["mean_completed_fraction"],
            clearance=metrics.get("clearance_p01_m", metrics["min_clearance_m"]),
        ),
        flush=True,
    )


def parse_task_weights(items: list[str]) -> dict[str, float]:
    weights = {}
    for item in items:
        if "=" not in item:
            raise SystemExit("--task-weight must be TASK=WEIGHT")
        task, value = item.split("=", 1)
        weights[task] = float(value)
    return weights


if __name__ == "__main__":
    main()
