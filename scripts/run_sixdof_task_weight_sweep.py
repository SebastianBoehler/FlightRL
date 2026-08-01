from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

from flightrl.sixdof.sweep import fmt, load_suite_summary, run_commands, status, sweep_summary


ROOT = Path(__file__).resolve().parents[1]
TASKS = "position_yaw,obstacle_avoidance,circle"


@dataclass(slots=True)
class WeightVariant:
    name: str
    task_weights: tuple[tuple[str, float], ...]
    hidden_size: int = 128
    epochs: int = 2
    learning_rate: float = 8e-4


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run 6-DoF multi-task offline task-weight sweeps")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="artifacts/task_weight_sweep/safe_tasks")
    parser.add_argument("--report", default="artifacts/replay/sixdof_task_weight_sweep.json")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--native-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-steps", type=int, default=80)
    parser.add_argument("--eval-num-envs", type=int, default=64)
    parser.add_argument("--suite-steps", type=int, default=300)
    parser.add_argument("--suite-num-envs", type=int, default=128)
    parser.add_argument("--baseline-checkpoint", default=None, help="Optional existing checkpoint to evaluate before trained variants.")
    args = parser.parse_args()

    variants = default_variants()
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    records = baseline_records(args) + [variant_record(args, variant) for variant in variants]
    if args.run:
        for record in records:
            record["results"] = run_commands(record["commands"], cwd=ROOT)
            record["gate"] = load_suite_summary(record["suite"])
    report = {"run": args.run, "records": records, "summary": sweep_summary(records)}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"summary={output}")
    print(f"markdown={output.with_suffix('.md')}")


def default_variants() -> list[WeightVariant]:
    return [
        WeightVariant("balanced_control", ()),
        WeightVariant("focus_position_circle_15", (("position_yaw", 1.5), ("circle", 1.5))),
        WeightVariant("focus_position_circle_2", (("position_yaw", 2.0), ("circle", 2.0))),
        WeightVariant("focus_circle_2", (("circle", 2.0),)),
        WeightVariant("focus_position_2", (("position_yaw", 2.0),)),
        WeightVariant("focus_position_circle_h256", (("position_yaw", 1.5), ("circle", 1.5)), hidden_size=256, epochs=3, learning_rate=7e-4),
    ]


def variant_record(args: argparse.Namespace, variant: WeightVariant) -> dict:
    base = Path(args.output_dir) / variant.name
    checkpoint = base / "checkpoint.pt"
    suite = base / "suite.json"
    commands = [train_command(args, variant, checkpoint), suite_command(args, variant, checkpoint, suite)]
    return {"variant": asdict(variant), "checkpoint": str(checkpoint), "suite": str(suite), "commands": commands}


def baseline_records(args: argparse.Namespace) -> list[dict]:
    if not args.baseline_checkpoint:
        return []
    base = Path(args.output_dir) / "baseline"
    suite = base / "suite.json"
    variant = {"name": "baseline", "task_weights": (), "hidden_size": None, "epochs": 0, "learning_rate": None}
    return [{"variant": variant, "checkpoint": args.baseline_checkpoint, "suite": str(suite), "commands": [suite_command(args, WeightVariant("baseline", ()), Path(args.baseline_checkpoint), suite)]}]


def train_command(args: argparse.Namespace, variant: WeightVariant, checkpoint: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_sixdof_offline.py"),
        "--dataset",
        args.dataset,
        "--checkpoint",
        str(checkpoint),
        "--epochs",
        str(variant.epochs),
        "--batch-size",
        "8192",
        "--hidden-size",
        str(variant.hidden_size),
        "--learning-rate",
        str(variant.learning_rate),
        "--eval-steps",
        str(args.eval_steps),
        "--eval-num-envs",
        str(args.eval_num_envs),
        "--select-by-eval",
    ]
    for task, weight in variant.task_weights:
        command.extend(["--task-weight", f"{task}={weight}"])
    if args.native_step:
        command.append("--native-step")
    return command


def suite_command(args: argparse.Namespace, variant: WeightVariant, checkpoint: Path, suite: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_sixdof_suite.py"),
        "--candidate",
        variant.name,
        str(checkpoint),
        "checkpoint",
        "--steps",
        str(args.suite_steps),
        "--num-envs",
        str(args.suite_num_envs),
        "--output",
        str(suite),
    ]
    if args.native_step:
        command.append("--native-step")
    return command


def render_markdown(report: dict) -> str:
    lines = ["# 6-DoF Task-Weight Sweep", "", "| variant | weights | status | completed | pos err m | clearance p01 m |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for record in report["records"]:
        gate = record.get("gate") or {}
        weights = ", ".join(f"{task}={weight}" for task, weight in record["variant"]["task_weights"]) or "none"
        lines.append(
            f"| {record['variant']['name']} | {weights} | {status(record)} | {fmt(gate.get('mean_completed_fraction'))} | "
            f"{fmt(gate.get('mean_position_error_m'))} | {fmt(gate.get('clearance_p01_m'))} |"
        )
    best = report["summary"].get("best")
    if best:
        lines.extend(["", f"Best by gate score: `{best['name']}` passed=`{best['passed']}` completed=`{best['mean_completed_fraction']:.4f}`."])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
