from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

from flightrl.sixdof.sweep import all_success, run_commands

ROOT = Path(__file__).resolve().parents[1]

@dataclass(slots=True)
class PpoVariant:
    name: str
    learning_rate: float
    action_std: float
    imitation_coef: float
    reference_coef: float
    reward_mode: str = "env"
    reset_profile: str = "position_yaw_medium"
    eval_reset_profile: str = "position_yaw_medium"
    updates: int = 16
    update_epochs: int = 2

def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run 6-DoF position/yaw PPO tuning sweeps")
    parser.add_argument("--init-checkpoint", default="artifacts/curriculum/position_yaw/easy_medium_h128/checkpoint.pt")
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--output-dir", default="artifacts/ppo/position_yaw")
    parser.add_argument("--report", default="artifacts/replay/sixdof_position_yaw_ppo_sweep.json")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--task", default="position_yaw")
    parser.add_argument("--train-num-envs", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--train-eval-steps", type=int, default=400)
    parser.add_argument("--eval-num-envs", type=int, default=256)
    parser.add_argument("--medium-steps", type=int, default=400)
    parser.add_argument("--broad-steps", type=int, default=800)
    parser.add_argument("--native-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-yaw-error-rad", type=float, default=0.35)
    parser.add_argument("--max-yaw-p95-error-rad", type=float, default=0.60)
    args = parser.parse_args()

    variants = default_variants()
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    records = baseline_records(args) + [variant_record(args, variant) for variant in variants]
    if args.run:
        for record in records:
            record["results"] = run_commands(record["commands"], cwd=ROOT)
            record["gates"] = load_gate_summaries(record)
    report = {"run": args.run, "thresholds": yaw_thresholds(args), "records": records, "summary": sweep_summary(records)}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"summary={output}")
    print(f"markdown={output.with_suffix('.md')}")

def default_variants() -> list[PpoVariant]:
    return [
        PpoVariant(name="stable_ref4_std002_lr1e5", learning_rate=1e-5, action_std=0.02, imitation_coef=0.20, reference_coef=4.0, reward_mode="progress_clearance", updates=12),
        PpoVariant(name="stable_ref8_std001_lr5e6", learning_rate=5e-6, action_std=0.01, imitation_coef=0.30, reference_coef=8.0, reward_mode="progress_clearance", updates=12),
        PpoVariant(name="ref1_std006_lr5e5", learning_rate=5e-5, action_std=0.06, imitation_coef=0.05, reference_coef=1.0),
        PpoVariant(name="ref2_std006_lr5e5", learning_rate=5e-5, action_std=0.06, imitation_coef=0.05, reference_coef=2.0),
        PpoVariant(name="progress_ref1_std006", learning_rate=5e-5, action_std=0.06, imitation_coef=0.05, reference_coef=1.0, reward_mode="progress"),
        PpoVariant(name="progress_ref2_std004", learning_rate=3e-5, action_std=0.04, imitation_coef=0.05, reference_coef=2.0, reward_mode="progress"),
        PpoVariant(
            name="broad_clearance_ref2_std006",
            learning_rate=3e-5,
            action_std=0.06,
            imitation_coef=0.05,
            reference_coef=2.0,
            reward_mode="progress_clearance",
            reset_profile="broad",
            eval_reset_profile="broad",
        ),
        PpoVariant(
            name="broad_clearance_ref1_std004",
            learning_rate=2e-5,
            action_std=0.04,
            imitation_coef=0.10,
            reference_coef=1.0,
            reward_mode="progress_clearance",
            reset_profile="broad",
            eval_reset_profile="broad",
        ),
    ]

def variant_record(args: argparse.Namespace, variant: PpoVariant) -> dict:
    base = Path(args.output_dir) / variant.name
    checkpoint = base / "checkpoint.pt"
    medium_gate = base / "medium_gate.json"
    broad_gate = base / "broad_gate.json"
    commands = [
        train_command(args, checkpoint, variant),
        eval_command(checkpoint, medium_gate, "position_yaw_medium", args.medium_steps, args),
        eval_command(checkpoint, broad_gate, "broad", args.broad_steps, args),
    ]
    return {
        "variant": asdict(variant),
        "checkpoint": str(checkpoint),
        "medium_gate": str(medium_gate),
        "broad_gate": str(broad_gate),
        "commands": commands,
    }

def baseline_records(args: argparse.Namespace) -> list[dict]:
    if not args.baseline_checkpoint:
        return []
    base = Path(args.output_dir) / "baseline"
    medium_gate = base / "medium_gate.json"
    broad_gate = base / "broad_gate.json"
    checkpoint = Path(args.baseline_checkpoint)
    variant = {"name": "baseline", "learning_rate": None, "action_std": None, "imitation_coef": None, "reference_coef": None}
    return [
        {
            "variant": variant,
            "checkpoint": str(checkpoint),
            "medium_gate": str(medium_gate),
            "broad_gate": str(broad_gate),
            "commands": [
                eval_command(checkpoint, medium_gate, "position_yaw_medium", args.medium_steps, args),
                eval_command(checkpoint, broad_gate, "broad", args.broad_steps, args),
            ],
        }
    ]

def train_command(args: argparse.Namespace, checkpoint: Path, variant: PpoVariant) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_sixdof_ppo.py"),
        "--init-checkpoint",
        args.init_checkpoint,
        "--checkpoint",
        str(checkpoint),
        "--task",
        args.task,
        "--updates",
        str(variant.updates),
        "--num-envs",
        str(args.train_num_envs),
        "--horizon",
        str(args.horizon),
        "--hidden-size",
        "128",
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
    ]
    if args.native_step:
        command.append("--native-step")
    return command


def eval_command(checkpoint: Path, output: Path, reset_profile: str, steps: int, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_sixdof_checkpoint.py"),
        "--checkpoint",
        str(checkpoint),
        "--task",
        args.task,
        "--steps",
        str(steps),
        "--num-envs",
        str(args.eval_num_envs),
        "--reset-profile",
        reset_profile,
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


def yaw_thresholds(args: argparse.Namespace) -> dict:
    return {"max_yaw_error_rad": args.max_yaw_error_rad, "max_yaw_p95_error_rad": args.max_yaw_p95_error_rad}


def load_gate_summaries(record: dict) -> dict:
    return {name: load_gate_summary(record[key]) for name, key in (("medium", "medium_gate"), ("broad", "broad_gate"))}


def load_gate_summary(path: str) -> dict | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    metrics = report["metrics"]
    return {
        "passed": report["gate"]["passed"],
        "failures": report["gate"]["failures"],
        "mean_completed_fraction": metrics["mean_completed_fraction"],
        "mean_survival_fraction": metrics["mean_survival_fraction"],
        "mean_position_error_m": metrics["mean_position_error_m"],
        "mean_yaw_error_rad": metrics.get("mean_yaw_error_rad"),
        "yaw_error_p95_rad": metrics.get("yaw_error_p95_rad"),
        "clearance_p01_m": metrics["clearance_p01_m"],
    }


def sweep_summary(records: list[dict]) -> dict:
    return {
        "total": len(records),
        "completed": sum(1 for record in records if all_success(record.get("results"))),
        "best_medium": best_record(records, "medium"),
        "best_broad": best_record(records, "broad"),
    }


def best_record(records: list[dict], gate_name: str) -> dict | None:
    candidates = []
    for record in records:
        gate = (record.get("gates") or {}).get(gate_name)
        if gate:
            candidates.append((gate_score(gate), compact_record(record, gate)))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def gate_score(gate: dict) -> tuple:
    return (0 if gate["passed"] else 1, -gate["mean_completed_fraction"], -gate["mean_survival_fraction"], gate["mean_position_error_m"], gate.get("mean_yaw_error_rad") or 0.0)


def compact_record(record: dict, gate: dict) -> dict:
    return {
        "name": record["variant"]["name"],
        "checkpoint": record["checkpoint"],
        "passed": gate["passed"],
        "failures": gate["failures"],
        "mean_completed_fraction": gate["mean_completed_fraction"],
        "mean_survival_fraction": gate["mean_survival_fraction"],
        "mean_position_error_m": gate["mean_position_error_m"],
        "mean_yaw_error_rad": gate.get("mean_yaw_error_rad"),
        "yaw_error_p95_rad": gate.get("yaw_error_p95_rad"),
        "clearance_p01_m": gate["clearance_p01_m"],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# 6-DoF Position/Yaw PPO Sweep",
        "",
        "| variant | status | medium completed | broad completed |",
        "| --- | --- | ---: | ---: |",
    ]
    for record in report["records"]:
        status = "planned"
        if record.get("results"):
            status = "ok" if all_success(record["results"]) else "failed"
        gates = record.get("gates") or {}
        lines.append(f"| {record['variant']['name']} | {status} | {fmt(gates.get('medium'))} | {fmt(gates.get('broad'))} |")
    summary = report.get("summary") or {}
    if summary.get("best_medium") or summary.get("best_broad"):
        lines.extend(["", "## Best Candidates", ""])
        for label, key in (("medium", "best_medium"), ("broad", "best_broad")):
            candidate = summary.get(key)
            if candidate:
                lines.append(
                    f"- `{label}`: `{candidate['name']}` passed=`{candidate['passed']}` "
                    f"completed=`{candidate['mean_completed_fraction']:.4f}` pos_err=`{candidate['mean_position_error_m']:.4f}`"
                )
    lines.extend(["", "Commands and artifact paths are stored in the JSON report."])
    return "\n".join(lines)


def fmt(gate: dict | None) -> str:
    return "pending" if gate is None else f"{gate['mean_completed_fraction']:.4f}"


if __name__ == "__main__":
    main()
