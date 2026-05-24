from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys

from flightrl.sixdof.sweep import all_success, run_commands


ROOT = Path(__file__).resolve().parents[1]
TASKS = "position_yaw,obstacle_avoidance,circle"
PROFILES = ("position_yaw_recovery", "broad")


@dataclass(slots=True)
class MultitaskPpoVariant:
    name: str
    task_probabilities: tuple[tuple[str, float], ...]
    learning_rate: float
    action_std: float
    imitation_coef: float
    reference_coef: float
    reward_mode: str = "progress_clearance"
    reset_profile: str = "broad"
    eval_reset_profile: str = "broad"
    updates: int = 8
    update_epochs: int = 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run profile-gated multitask 6-DoF PPO sweeps")
    parser.add_argument("--init-checkpoint", default="artifacts/dagger/sixdof_safe_tasks_horizon800/iter_01.pt")
    parser.add_argument("--baseline-checkpoint", default="artifacts/dagger/sixdof_safe_tasks_horizon800/iter_01.pt")
    parser.add_argument("--output-dir", default="artifacts/ppo/multitask_profile_sweep")
    parser.add_argument("--report", default="artifacts/replay/sixdof_multitask_ppo_sweep.json")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None, help="Override variant PPO updates for smoke runs.")
    parser.add_argument("--train-num-envs", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=None, help="Override policy width. Defaults to the init checkpoint width.")
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--train-eval-steps", type=int, default=300)
    parser.add_argument("--eval-num-envs", type=int, default=128)
    parser.add_argument("--suite-steps", type=int, default=360)
    parser.add_argument("--suite-num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=4217)
    parser.add_argument("--native-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-yaw-error-rad", type=float, default=0.35)
    parser.add_argument("--max-yaw-p95-error-rad", type=float, default=0.60)
    args = parser.parse_args()

    variants = default_variants()
    if args.updates is not None:
        variants = [replace(variant, updates=args.updates) for variant in variants]
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    records = baseline_records(args) + [variant_record(args, variant) for variant in variants]
    report = {"run": args.run, "records": records, "profile_matrix": profile_matrix_path(args), "commands": validation_commands(args, records)}
    if args.run:
        report["results"] = run_commands(flatten(record["commands"] for record in records) + report["commands"], cwd=ROOT)
        report["profile_summary"] = load_profile_summary(Path(report["profile_matrix"]))
    report["summary"] = summarize(report)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"summary={output}")
    print(f"markdown={output.with_suffix('.md')}")


def default_variants() -> list[MultitaskPpoVariant]:
    return [
        MultitaskPpoVariant("balanced_h64_ref2_std002", (), 2e-5, 0.02, 0.10, 2.0),
        MultitaskPpoVariant("py_focus4_h64_ref2_std002", (("position_yaw", 4.0),), 2e-5, 0.02, 0.10, 2.0),
        MultitaskPpoVariant("py_yaw_focus4_h64_ref2_std002", (("position_yaw", 4.0),), 2e-5, 0.02, 0.10, 2.0, reward_mode="progress_yaw_clearance"),
        MultitaskPpoVariant("py_circle3_h64_ref2_std002", (("position_yaw", 3.0), ("circle", 3.0)), 2e-5, 0.02, 0.10, 2.0),
        MultitaskPpoVariant("yaw_conservative_ref4_std001", (("position_yaw", 4.0),), 1e-5, 0.01, 0.20, 4.0, reward_mode="progress_yaw_clearance", updates=10),
        MultitaskPpoVariant("conservative_ref4_std001", (("position_yaw", 4.0),), 1e-5, 0.01, 0.20, 4.0, updates=10),
    ]


def baseline_records(args: argparse.Namespace) -> list[dict]:
    if not args.baseline_checkpoint:
        return []
    return [{"variant": {"name": "baseline", "task_probabilities": (), "updates": 0}, "checkpoint": args.baseline_checkpoint, "commands": []}]


def variant_record(args: argparse.Namespace, variant: MultitaskPpoVariant) -> dict:
    checkpoint = Path(args.output_dir) / variant.name / "checkpoint.pt"
    return {"variant": asdict(variant), "checkpoint": str(checkpoint), "commands": [train_command(args, variant, checkpoint)]}


def train_command(args: argparse.Namespace, variant: MultitaskPpoVariant, checkpoint: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_sixdof_ppo.py"),
        "--init-checkpoint",
        args.init_checkpoint,
        "--checkpoint",
        str(checkpoint),
        "--task",
        TASKS,
        "--train-tasks",
        TASKS,
        "--updates",
        str(variant.updates),
        "--num-envs",
        str(args.train_num_envs),
        "--horizon",
        str(args.horizon),
        "--learning-rate",
        str(variant.learning_rate),
        "--update-epochs",
        str(variant.update_epochs),
        "--minibatch-size",
        str(args.minibatch_size),
        "--action-std",
        str(variant.action_std),
        "--imitation-coef",
        str(variant.imitation_coef),
        "--reference-coef",
        str(variant.reference_coef),
        "--reward-mode",
        variant.reward_mode,
        "--reset-profile",
        variant.reset_profile,
        "--eval-reset-profile",
        variant.eval_reset_profile,
        "--eval-steps",
        str(args.train_eval_steps),
        "--eval-num-envs",
        str(args.eval_num_envs),
    ]
    if args.hidden_size is not None:
        command.extend(["--hidden-size", str(args.hidden_size)])
    for task, weight in variant.task_probabilities:
        command.extend(["--task-probability", f"{task}={weight}"])
    if args.native_step:
        command.append("--native-step")
    return command


def validation_commands(args: argparse.Namespace, records: list[dict]) -> list[list[str]]:
    commands = [suite_command(args, records, profile, profile_suite_path(args, profile)) for profile in PROFILES]
    commands.append(matrix_command(args))
    return commands


def suite_command(args: argparse.Namespace, records: list[dict], profile: str, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_sixdof_suite.py"),
        "--steps",
        str(args.suite_steps),
        "--num-envs",
        str(args.suite_num_envs),
        "--seed",
        str(args.seed + PROFILES.index(profile)),
        "--reset-profile",
        profile,
        "--max-yaw-error-rad",
        str(args.max_yaw_error_rad),
        "--max-yaw-p95-error-rad",
        str(args.max_yaw_p95_error_rad),
        "--output",
        str(output),
    ]
    if args.native_step:
        command.append("--native-step")
    for record in records:
        command.extend(["--candidate", record["variant"]["name"], record["checkpoint"], "checkpoint"])
    return command


def matrix_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(ROOT / "scripts" / "build_sixdof_profile_matrix.py")]
    for profile in PROFILES:
        command.extend(["--suite", str(profile_suite_path(args, profile))])
    command.extend(["--output", profile_matrix_path(args)])
    return command


def profile_suite_path(args: argparse.Namespace, profile: str) -> Path:
    return Path(args.output_dir) / f"profile_{profile}.json"


def profile_matrix_path(args: argparse.Namespace) -> str:
    return str(Path(args.output_dir) / "profile_matrix.json")


def load_profile_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("records", [])


def summarize(report: dict) -> dict:
    records = report.get("profile_summary") or []
    return {"total": len(report["records"]), "best": compact(records[0]) if records else None, "completed": all_success(report.get("results"))}


def compact(record: dict) -> dict:
    return {
        "label": record["label"],
        "checkpoint": record["checkpoint"],
        "passed_all_profiles": record["passed_all_profiles"],
        "worst_completed_fraction": record["worst_completed_fraction"],
        "worst_position_error_m": record["worst_position_error_m"],
        "worst_yaw_error_rad": record.get("worst_yaw_error_rad"),
        "worst_clearance_p01_m": record["worst_clearance_p01_m"],
    }


def render_markdown(report: dict) -> str:
    lines = ["# 6-DoF Multitask PPO Sweep", "", "| variant | probabilities | status |", "| --- | --- | --- |"]
    for record in report["records"]:
        probabilities = ", ".join(f"{task}={weight}" for task, weight in record["variant"]["task_probabilities"]) or "uniform"
        lines.append(f"| {record['variant']['name']} | {probabilities} | {status(report)} |")
    if report.get("profile_summary"):
        lines.extend(["", "| candidate | all passed | worst completed | worst pos err m | worst yaw rad | worst clearance m |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for record in report["profile_summary"]:
            lines.append(
                f"| {record['label']} | {record['passed_all_profiles']} | {fmt(record['worst_completed_fraction'])} | "
                f"{fmt(record['worst_position_error_m'])} | {fmt(record.get('worst_yaw_error_rad'))} | {fmt(record['worst_clearance_p01_m'])} |"
            )
    best = report.get("summary", {}).get("best")
    if best:
        lines.extend(["", f"Best by profile matrix: `{best['label']}` all_passed=`{best['passed_all_profiles']}` completed=`{best['worst_completed_fraction']:.4f}`."])
    lines.extend(["", "Simulation-only sweep; this does not approve live Crazyflie deployment."])
    return "\n".join(lines)


def status(report: dict) -> str:
    if "results" not in report:
        return "planned"
    return "ok" if all_success(report["results"]) else "failed"


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def flatten(groups) -> list[list[str]]:
    return [command for group in groups for command in group]


if __name__ == "__main__":
    main()
