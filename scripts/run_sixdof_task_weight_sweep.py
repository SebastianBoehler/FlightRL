from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


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
    parser.add_argument("--dataset", default="artifacts/dagger/sixdof_safe_tasks_horizon800/iter_01.npz")
    parser.add_argument("--output-dir", default="artifacts/task_weight_sweep/safe_tasks")
    parser.add_argument("--report", default="artifacts/replay/sixdof_task_weight_sweep.json")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--native-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-steps", type=int, default=80)
    parser.add_argument("--eval-num-envs", type=int, default=64)
    parser.add_argument("--suite-steps", type=int, default=300)
    parser.add_argument("--suite-num-envs", type=int, default=128)
    args = parser.parse_args()

    variants = default_variants()
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    records = [variant_record(args, variant) for variant in variants]
    if args.run:
        for record in records:
            record["results"] = run_commands(record["commands"])
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


def run_commands(commands: list[list[str]]) -> list[dict]:
    results = []
    for command in commands:
        start = perf_counter()
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        results.append({"command": command, "returncode": completed.returncode, "elapsed_s": perf_counter() - start})
        if completed.returncode != 0:
            break
    return results


def load_suite_summary(path: str) -> dict | None:
    suite = Path(path)
    if not suite.exists():
        return None
    record = json.loads(suite.read_text())["records"][0]
    metrics = record["metrics"]
    return {
        "passed": record["gate"]["passed"],
        "failures": record["gate"]["failures"],
        "mean_completed_fraction": metrics["mean_completed_fraction"],
        "mean_survival_fraction": metrics["mean_survival_fraction"],
        "mean_position_error_m": metrics["mean_position_error_m"],
        "clearance_p01_m": metrics["clearance_p01_m"],
        "per_task_gate": record.get("per_task_gate", {}),
    }


def sweep_summary(records: list[dict]) -> dict:
    return {"total": len(records), "completed": sum(1 for record in records if all_success(record.get("results"))), "best": best_record(records)}


def all_success(results: list[dict] | None) -> bool:
    return bool(results) and all(result["returncode"] == 0 for result in results)


def best_record(records: list[dict]) -> dict | None:
    candidates = [(gate_score(record["gate"]), compact_record(record)) for record in records if record.get("gate")]
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def gate_score(gate: dict) -> tuple:
    return (0 if gate["passed"] else 1, -gate["mean_completed_fraction"], -gate["mean_survival_fraction"], gate["mean_position_error_m"], -gate["clearance_p01_m"])


def compact_record(record: dict) -> dict:
    gate = record["gate"]
    return {"name": record["variant"]["name"], "checkpoint": record["checkpoint"], **gate}


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


def status(record: dict) -> str:
    if "results" not in record:
        return "planned"
    return "ok" if all_success(record["results"]) else "failed"


def fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "pending"


if __name__ == "__main__":
    main()
