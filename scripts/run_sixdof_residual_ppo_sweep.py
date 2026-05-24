from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys

from flightrl.sixdof.sweep import all_success, run_commands


ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ResidualPpoVariant:
    name: str
    residual_scale: float
    learning_rate: float
    action_std: float
    imitation_coef: float
    reference_coef: float
    updates: int = 8
    reward_mode: str = "progress_yaw_clearance"
    reset_profile: str = "circle_recovery"
    eval_reset_profile: str = "circle_recovery"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run teacher-residual 6-DoF PPO sweeps")
    parser.add_argument("--output-dir", default="artifacts/ppo/circle_residual_sweep")
    parser.add_argument("--report", default="artifacts/replay/sixdof_circle_residual_ppo_sweep.json")
    parser.add_argument("--task", default="circle")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--train-num-envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--train-eval-steps", type=int, default=300)
    parser.add_argument("--gate-steps", type=int, default=300)
    parser.add_argument("--eval-num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--native-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-yaw-error-rad", type=float, default=0.60)
    parser.add_argument("--max-yaw-p95-error-rad", type=float, default=1.00)
    parser.add_argument("--max-teacher-action-l2-mean", type=float, default=0.02)
    args = parser.parse_args()

    variants = default_variants()
    if args.updates is not None:
        variants = [replace(variant, updates=args.updates) for variant in variants]
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    records = [variant_record(args, variant) for variant in variants]
    if args.run:
        for record in records:
            record["results"] = run_commands(record["commands"], cwd=ROOT)
            record["gate"] = load_gate_summary(record["gate_path"], args.max_teacher_action_l2_mean)
    report = {"run": args.run, "records": records, "thresholds": thresholds(args), "summary": sweep_summary(records)}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"summary={output}")
    print(f"markdown={output.with_suffix('.md')}")


def default_variants() -> list[ResidualPpoVariant]:
    return [
        ResidualPpoVariant("scale005_ref4_std001", 0.05, 1e-5, 0.01, 0.20, 4.0),
        ResidualPpoVariant("scale010_ref2_std002", 0.10, 1e-5, 0.02, 0.10, 2.0),
        ResidualPpoVariant("scale015_ref2_std002", 0.15, 1e-5, 0.02, 0.10, 2.0),
        ResidualPpoVariant("scale020_ref4_std001", 0.20, 5e-6, 0.01, 0.20, 4.0, updates=10),
    ]


def variant_record(args: argparse.Namespace, variant: ResidualPpoVariant) -> dict:
    base = Path(args.output_dir) / variant.name
    scaffold = base / "residual_scaffold.pt"
    checkpoint = base / "checkpoint.pt"
    gate = base / "circle_recovery_gate.json"
    return {
        "variant": asdict(variant),
        "scaffold": str(scaffold),
        "checkpoint": str(checkpoint),
        "gate_path": str(gate),
        "commands": [scaffold_command(args, variant, scaffold), train_command(args, variant, scaffold, checkpoint), gate_command(args, checkpoint, gate)],
    }


def scaffold_command(args: argparse.Namespace, variant: ResidualPpoVariant, checkpoint: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "create_sixdof_residual_checkpoint.py"),
        "--checkpoint",
        str(checkpoint),
        "--task",
        args.task,
        "--hidden-size",
        str(args.hidden_size),
        "--residual-scale",
        str(variant.residual_scale),
        "--zero-weights",
        "--seed",
        str(args.seed),
    ]


def train_command(args: argparse.Namespace, variant: ResidualPpoVariant, scaffold: Path, checkpoint: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_sixdof_ppo.py"),
        "--init-checkpoint",
        str(scaffold),
        "--checkpoint",
        str(checkpoint),
        "--task",
        args.task,
        "--controller",
        "teacher_residual",
        "--residual-scale",
        str(variant.residual_scale),
        "--updates",
        str(variant.updates),
        "--num-envs",
        str(args.train_num_envs),
        "--horizon",
        str(args.horizon),
        "--hidden-size",
        str(args.hidden_size),
        "--learning-rate",
        str(variant.learning_rate),
        "--update-epochs",
        "2",
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
        "--max-yaw-error-rad",
        str(args.max_yaw_error_rad),
        "--max-yaw-p95-error-rad",
        str(args.max_yaw_p95_error_rad),
    ]
    if args.native_step:
        command.append("--native-step")
    return command


def gate_command(args: argparse.Namespace, checkpoint: Path, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_sixdof_checkpoint.py"),
        "--checkpoint",
        str(checkpoint),
        "--task",
        args.task,
        "--steps",
        str(args.gate_steps),
        "--num-envs",
        str(args.eval_num_envs),
        "--reset-profile",
        "circle_recovery",
        "--output",
        str(output),
        "--max-yaw-error-rad",
        str(args.max_yaw_error_rad),
        "--max-yaw-p95-error-rad",
        str(args.max_yaw_p95_error_rad),
    ]
    if args.native_step:
        command.append("--native-step")
    return command


def load_gate_summary(path: str, max_teacher_l2: float) -> dict | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    metrics = report["metrics"]
    teacher_l2 = float(metrics.get("teacher_action_l2_mean", 0.0))
    return {
        "passed": bool(report["gate"]["passed"] and teacher_l2 <= max_teacher_l2),
        "sim_gate_passed": report["gate"]["passed"],
        "failures": report["gate"]["failures"] + ([] if teacher_l2 <= max_teacher_l2 else ["teacher_action_l2"]),
        "mean_completed_fraction": metrics["mean_completed_fraction"],
        "mean_survival_fraction": metrics["mean_survival_fraction"],
        "mean_position_error_m": metrics["mean_position_error_m"],
        "mean_yaw_error_rad": metrics.get("mean_yaw_error_rad"),
        "yaw_error_p95_rad": metrics.get("yaw_error_p95_rad"),
        "clearance_p01_m": metrics["clearance_p01_m"],
        "teacher_action_l2_mean": teacher_l2,
        "teacher_action_l2_p95": metrics.get("teacher_action_l2_p95"),
    }


def thresholds(args: argparse.Namespace) -> dict:
    return {
        "max_yaw_error_rad": args.max_yaw_error_rad,
        "max_yaw_p95_error_rad": args.max_yaw_p95_error_rad,
        "max_teacher_action_l2_mean": args.max_teacher_action_l2_mean,
    }


def sweep_summary(records: list[dict]) -> dict:
    return {"total": len(records), "completed": sum(1 for record in records if all_success(record.get("results"))), "best": best_record(records)}


def best_record(records: list[dict]) -> dict | None:
    candidates = [(gate_score(record["gate"]), compact_record(record)) for record in records if record.get("gate")]
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def gate_score(gate: dict) -> tuple:
    return (0 if gate["passed"] else 1, -gate["mean_completed_fraction"], gate["mean_position_error_m"], gate["mean_yaw_error_rad"] or 0.0, gate["teacher_action_l2_mean"])


def compact_record(record: dict) -> dict:
    return {"name": record["variant"]["name"], "checkpoint": record["checkpoint"], **record["gate"]}


def render_markdown(report: dict) -> str:
    lines = ["# 6-DoF Circle Residual PPO Sweep", "", "| variant | status | completed | orbit err m | yaw rad | teacher l2 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for record in report["records"]:
        gate = record.get("gate")
        lines.append(f"| {record['variant']['name']} | {status(record)} | {fmt(gate, 'mean_completed_fraction')} | {fmt(gate, 'mean_position_error_m')} | {fmt(gate, 'mean_yaw_error_rad')} | {fmt(gate, 'teacher_action_l2_mean')} |")
    best = report.get("summary", {}).get("best")
    if best:
        lines.extend(["", f"Best residual candidate: `{best['name']}` passed=`{best['passed']}` teacher_l2=`{best['teacher_action_l2_mean']:.6f}`."])
    lines.extend(["", "Simulation-only sweep; this does not approve live Crazyflie deployment."])
    return "\n".join(lines)


def status(record: dict) -> str:
    if "results" not in record:
        return "planned"
    return "ok" if all_success(record["results"]) else "failed"


def fmt(gate: dict | None, key: str) -> str:
    return "pending" if gate is None or gate.get(key) is None else f"{gate[key]:.4f}"


if __name__ == "__main__":
    main()
