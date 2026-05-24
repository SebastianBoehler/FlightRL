from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

from flightrl.sixdof.sweep import all_success, run_commands


ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class CircleVariant:
    name: str
    profiles: tuple[str, ...]
    num_envs: int
    steps_per_profile: int
    epochs: int
    hidden_size: int
    learning_rate: float
    eval_profile: str
    observation_mode: str = "base"
    action_weighting: str = "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run circle-specific 6-DoF curriculum sweeps")
    parser.add_argument("--output-dir", default="artifacts/curriculum/circle")
    parser.add_argument("--report", default="artifacts/replay/sixdof_circle_curriculum_sweep.json")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--native-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=None, help="Override steps per profile for smoke runs.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for smoke runs.")
    parser.add_argument("--suite-steps", type=int, default=300)
    parser.add_argument("--suite-num-envs", type=int, default=128)
    parser.add_argument("--max-yaw-error-rad", type=float, default=0.35)
    parser.add_argument("--max-yaw-p95-error-rad", type=float, default=0.60)
    args = parser.parse_args()

    variants = override_variants(default_variants(), args)
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    records = [variant_record(args, variant) for variant in variants]
    if args.run:
        for record in records:
            record["results"] = run_commands(record["commands"], cwd=ROOT)
            record["gates"] = load_gates(record)
    report = {"run": args.run, "records": records, "summary": summarize(records)}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"summary={output}")
    print(f"markdown={output.with_suffix('.md')}")


def default_variants() -> list[CircleVariant]:
    return [
        CircleVariant("easy_recovery_h128", ("circle_easy", "circle_recovery"), 512, 192, 12, 128, 1e-3, "circle_recovery"),
        CircleVariant("easy_recovery_h256", ("circle_easy", "circle_recovery"), 512, 192, 12, 256, 8e-4, "circle_recovery"),
        CircleVariant("recovery_history1_h128", ("circle_recovery",), 512, 256, 14, 128, 8e-4, "circle_recovery", "history1"),
        CircleVariant("recovery_weighted_h256", ("circle_recovery",), 768, 256, 16, 256, 6e-4, "circle_recovery", action_weighting="inverse_std"),
    ]


def override_variants(variants: list[CircleVariant], args: argparse.Namespace) -> list[CircleVariant]:
    updated = []
    for variant in variants:
        data = asdict(variant)
        if args.steps is not None:
            data["steps_per_profile"] = args.steps
        if args.epochs is not None:
            data["epochs"] = args.epochs
        updated.append(CircleVariant(**data))
    return updated


def variant_record(args: argparse.Namespace, variant: CircleVariant) -> dict:
    base = Path(args.output_dir) / variant.name
    dataset = dataset_commands(base, variant, args.native_step)
    checkpoint = base / "checkpoint.pt"
    recovery_gate = base / "circle_recovery_gate.json"
    broad_gate = base / "broad_gate.json"
    commands = dataset["commands"] + [
        train_command(dataset["final"], checkpoint, variant, args.native_step),
        eval_command(args, checkpoint, recovery_gate, "circle_recovery"),
        eval_command(args, checkpoint, broad_gate, "broad"),
    ]
    return {"variant": asdict(variant), "dataset": str(dataset["final"]), "checkpoint": str(checkpoint), "recovery_gate": str(recovery_gate), "broad_gate": str(broad_gate), "commands": commands}


def dataset_commands(base: Path, variant: CircleVariant, native_step: bool) -> dict:
    commands = []
    previous: Path | None = None
    for index, profile in enumerate(variant.profiles, start=1):
        dataset = base / f"dataset_{index:02d}_{profile}.npz"
        command = [sys.executable, str(ROOT / "scripts" / "build_sixdof_teacher_dataset.py"), "--task", "circle", "--num-envs", str(variant.num_envs), "--steps", str(variant.steps_per_profile), "--seed", str(1300 + index), "--reset-profile", profile, "--observation-mode", variant.observation_mode, "--output", str(dataset)]
        if native_step:
            command.append("--native-step")
        if previous is not None:
            command.extend(["--append-dataset", str(previous)])
        commands.append(command)
        previous = dataset
    assert previous is not None
    return {"commands": commands, "final": previous}


def train_command(dataset: Path, checkpoint: Path, variant: CircleVariant, native_step: bool) -> list[str]:
    command = [sys.executable, str(ROOT / "scripts" / "train_sixdof_offline.py"), "--dataset", str(dataset), "--checkpoint", str(checkpoint), "--epochs", str(variant.epochs), "--hidden-size", str(variant.hidden_size), "--learning-rate", str(variant.learning_rate), "--eval-reset-profile", variant.eval_profile, "--eval-steps", "300", "--select-by-eval"]
    if variant.action_weighting != "none":
        command.extend(["--action-weighting", variant.action_weighting])
    if native_step:
        command.append("--native-step")
    return command


def eval_command(args: argparse.Namespace, checkpoint: Path, output: Path, profile: str) -> list[str]:
    command = [sys.executable, str(ROOT / "scripts" / "evaluate_sixdof_checkpoint.py"), "--checkpoint", str(checkpoint), "--task", "circle", "--steps", str(args.suite_steps), "--num-envs", str(args.suite_num_envs), "--reset-profile", profile, "--output", str(output), "--max-yaw-error-rad", str(args.max_yaw_error_rad), "--max-yaw-p95-error-rad", str(args.max_yaw_p95_error_rad)]
    if args.native_step:
        command.append("--native-step")
    return command


def load_gates(record: dict) -> dict:
    return {name: load_gate(record[key]) for name, key in (("recovery", "recovery_gate"), ("broad", "broad_gate"))}


def load_gate(path: str) -> dict | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    metrics = report["metrics"]
    return {"passed": report["gate"]["passed"], "failures": report["gate"]["failures"], "completed": metrics["mean_completed_fraction"], "position_error": metrics["mean_position_error_m"], "yaw_p95": metrics.get("yaw_error_p95_rad"), "clearance": metrics["clearance_p01_m"]}


def summarize(records: list[dict]) -> dict:
    candidates = [(score(record), record) for record in records if record.get("gates")]
    best = min(candidates, default=(None, None))[1]
    return {"total": len(records), "completed": sum(1 for record in records if all_success(record.get("results"))), "best": compact(best) if best else None}


def score(record: dict) -> tuple:
    gate = record["gates"]["recovery"]
    return (0 if gate["passed"] else 1, -gate["completed"], gate["position_error"], gate["yaw_p95"] or 999.0, -gate["clearance"])


def compact(record: dict) -> dict:
    return {"name": record["variant"]["name"], "checkpoint": record["checkpoint"], **record["gates"]["recovery"]}


def render_markdown(report: dict) -> str:
    lines = ["# 6-DoF Circle Curriculum Sweep", "", "| variant | profiles | status | recovery completed | recovery pos err | recovery yaw p95 | broad completed |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for record in report["records"]:
        gates = record.get("gates") or {}
        lines.append(f"| {record['variant']['name']} | {', '.join(record['variant']['profiles'])} | {status(record)} | {fmt(gates.get('recovery'), 'completed')} | {fmt(gates.get('recovery'), 'position_error')} | {fmt(gates.get('recovery'), 'yaw_p95')} | {fmt(gates.get('broad'), 'completed')} |")
    best = report.get("summary", {}).get("best")
    if best:
        lines.extend(["", f"Best recovery candidate: `{best['name']}` passed=`{best['passed']}` completed=`{best['completed']:.4f}` pos_err=`{best['position_error']:.4f}`."])
    lines.append("\nSimulation-only sweep; this does not approve live hardware.")
    return "\n".join(lines)


def status(record: dict) -> str:
    if "results" not in record:
        return "planned"
    return "ok" if all_success(record["results"]) else "failed"


def fmt(gate: dict | None, key: str) -> str:
    return "pending" if gate is None or gate.get(key) is None else f"{gate[key]:.4f}"


if __name__ == "__main__":
    main()
