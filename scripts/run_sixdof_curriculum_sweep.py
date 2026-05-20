from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class CurriculumVariant:
    name: str
    profiles: tuple[str, ...]
    num_envs: int
    steps_per_profile: int
    epochs: int
    hidden_size: int
    learning_rate: float
    eval_profile: str
    eval_steps: int
    observation_mode: str = "base"
    action_weighting: str = "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run staged 6-DoF position/yaw curriculum sweeps")
    parser.add_argument("--output-dir", default="artifacts/curriculum/position_yaw")
    parser.add_argument("--report", default="artifacts/replay/sixdof_position_yaw_curriculum_sweep.json")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--native-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-yaw-error-rad", type=float, default=0.35)
    parser.add_argument("--max-yaw-p95-error-rad", type=float, default=0.60)
    args = parser.parse_args()

    variants = default_variants()
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    records = [variant_record(args, variant) for variant in variants]
    if args.run:
        for record in records:
            record["results"] = run_commands(record["commands"])
            record["gates"] = load_gate_summaries(record)

    report = {"run": args.run, "native_step": args.native_step, "thresholds": yaw_thresholds(args), "records": records, "summary": sweep_summary(records)}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"summary={output}")
    print(f"markdown={output.with_suffix('.md')}")


def default_variants() -> list[CurriculumVariant]:
    return [
        CurriculumVariant("easy_medium_h128", ("position_yaw_easy", "position_yaw_medium"), 512, 192, 12, 128, 1e-3, "position_yaw_medium", 400),
        CurriculumVariant("easy_medium_h256", ("position_yaw_easy", "position_yaw_medium"), 512, 192, 12, 256, 1e-3, "position_yaw_medium", 400),
        CurriculumVariant("medium_h256_long", ("position_yaw_medium",), 1024, 384, 16, 256, 7e-4, "position_yaw_medium", 600),
        CurriculumVariant("easy_medium_history1_h128", ("position_yaw_easy", "position_yaw_medium"), 512, 192, 12, 128, 1e-3, "position_yaw_medium", 400, "history1"),
        CurriculumVariant("easy_medium_history1_inverse_std_h128", ("position_yaw_easy", "position_yaw_medium"), 512, 192, 12, 128, 1e-3, "position_yaw_medium", 400, "history1", "inverse_std"),
        CurriculumVariant("easy_medium_wide_h128", ("position_yaw_easy", "position_yaw_medium", "position_yaw_wide"), 512, 192, 14, 128, 8e-4, "position_yaw_wide", 500),
        CurriculumVariant("easy_medium_broad_h256", ("position_yaw_easy", "position_yaw_medium", "broad"), 512, 192, 16, 256, 7e-4, "broad", 600),
    ]


def variant_record(args: argparse.Namespace, variant: CurriculumVariant) -> dict:
    base = Path(args.output_dir) / variant.name
    dataset_paths = dataset_commands(base, variant, args.native_step)
    checkpoint = base / "checkpoint.pt"
    medium_gate = base / "medium_gate.json"
    broad_gate = base / "broad_gate.json"
    commands = dataset_paths["commands"]
    commands.append(train_command(dataset_paths["final"], checkpoint, variant, args.native_step))
    commands.append(eval_command(checkpoint, medium_gate, "position_yaw_medium", variant.eval_steps, args))
    commands.append(eval_command(checkpoint, broad_gate, "broad", 800, args))
    return {
        "variant": asdict(variant),
        "dataset": str(dataset_paths["final"]),
        "checkpoint": str(checkpoint),
        "medium_gate": str(medium_gate),
        "broad_gate": str(broad_gate),
        "commands": commands,
    }


def dataset_commands(base: Path, variant: CurriculumVariant, native_step: bool) -> dict:
    commands = []
    previous: Path | None = None
    for index, profile in enumerate(variant.profiles, start=1):
        dataset = base / f"dataset_{index:02d}_{profile}.npz"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "build_sixdof_teacher_dataset.py"),
            "--task",
            "position_yaw",
            "--num-envs",
            str(variant.num_envs),
            "--steps",
            str(variant.steps_per_profile),
            "--seed",
            str(700 + index),
            "--reset-profile",
            profile,
            "--observation-mode",
            variant.observation_mode,
            "--output",
            str(dataset),
        ]
        if native_step:
            command.append("--native-step")
        if previous is not None:
            command.extend(["--append-dataset", str(previous)])
        commands.append(command)
        previous = dataset
    assert previous is not None
    return {"commands": commands, "final": previous}


def train_command(dataset: Path, checkpoint: Path, variant: CurriculumVariant, native_step: bool) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_sixdof_offline.py"),
        "--dataset",
        str(dataset),
        "--checkpoint",
        str(checkpoint),
        "--epochs",
        str(variant.epochs),
        "--hidden-size",
        str(variant.hidden_size),
        "--learning-rate",
        str(variant.learning_rate),
        "--eval-steps",
        str(variant.eval_steps),
        "--select-by-eval",
        "--eval-reset-profile",
        variant.eval_profile,
    ]
    if variant.action_weighting != "none":
        command.extend(["--action-weighting", variant.action_weighting])
    if native_step:
        command.append("--native-step")
    return command


def eval_command(checkpoint: Path, output: Path, reset_profile: str, steps: int, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_sixdof_checkpoint.py"),
        "--checkpoint",
        str(checkpoint),
        "--task",
        "position_yaw",
        "--steps",
        str(steps),
        "--num-envs",
        "256",
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


def run_commands(commands: list[list[str]]) -> list[dict]:
    results = []
    for command in commands:
        start = perf_counter()
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        results.append({"command": command, "returncode": completed.returncode, "elapsed_s": perf_counter() - start})
        if completed.returncode != 0:
            break
    return results


def load_gate_summaries(record: dict) -> dict:
    return {name: load_gate_summary(record[path_key]) for name, path_key in (("medium", "medium_gate"), ("broad", "broad_gate"))}


def load_gate_summary(path: str) -> dict | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    metrics = report["metrics"]
    return {
        "passed": report["gate"]["passed"],
        "failures": report["gate"]["failures"],
        "mean_position_error_m": metrics["mean_position_error_m"],
        "mean_yaw_error_rad": metrics.get("mean_yaw_error_rad"),
        "yaw_error_p95_rad": metrics.get("yaw_error_p95_rad"),
        "clearance_p01_m": metrics["clearance_p01_m"],
        "mean_completed_fraction": metrics["mean_completed_fraction"],
        "mean_survival_fraction": metrics["mean_survival_fraction"],
    }


def sweep_summary(records: list[dict]) -> dict:
    return {
        "total": len(records),
        "completed": sum(1 for record in records if all_success(record.get("results"))),
        "best_medium": best_record(records, "medium"),
        "best_broad": best_record(records, "broad"),
    }


def all_success(results: list[dict] | None) -> bool:
    return bool(results) and all(result["returncode"] == 0 for result in results)


def best_record(records: list[dict], gate_name: str) -> dict | None:
    candidates = []
    for record in records:
        gate = (record.get("gates") or {}).get(gate_name)
        if gate is not None:
            candidates.append((gate_score(gate), compact_record(record, gate)))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def gate_score(gate: dict) -> tuple:
    return (0 if gate["passed"] else 1, -gate["mean_completed_fraction"], -gate["mean_survival_fraction"], gate["mean_position_error_m"], gate.get("mean_yaw_error_rad") or 0.0, -gate["clearance_p01_m"])


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
        "# 6-DoF Position/Yaw Curriculum Sweep",
        "",
        "| variant | profiles | hidden | weighting | epochs | eval profile | status | medium completed | broad completed |",
        "| --- | --- | ---: | --- | ---: | --- | --- | ---: | ---: |",
    ]
    for record in report["records"]:
        variant = record["variant"]
        results = record.get("results")
        status = "planned"
        if results:
            status = "ok" if all(result["returncode"] == 0 for result in results) else "failed"
        gates = record.get("gates") or {}
        medium_completed = format_completed(gates.get("medium"))
        broad_completed = format_completed(gates.get("broad"))
        lines.append(
            f"| {variant['name']} | {', '.join(variant['profiles'])} | {variant['hidden_size']} | "
            f"{variant.get('action_weighting', 'none')} | {variant['epochs']} | {variant['eval_profile']} | "
            f"{status} | {medium_completed} | {broad_completed} |"
        )
    summary = report.get("summary") or {}
    if summary.get("best_medium") or summary.get("best_broad"):
        lines.extend(["", "## Best Candidates", ""])
        for label, key in (("medium", "best_medium"), ("broad", "best_broad")):
            candidate = summary.get(key)
            if candidate:
                lines.append(
                    f"- `{label}`: `{candidate['name']}` passed=`{candidate['passed']}` "
                    f"completed=`{candidate['mean_completed_fraction']:.4f}` "
                    f"pos_err=`{candidate['mean_position_error_m']:.4f}`"
                )
    lines.extend(["", "Commands and artifact paths are stored in the JSON report."])
    return "\n".join(lines)


def format_completed(gate: dict | None) -> str:
    return "pending" if gate is None else f"{gate['mean_completed_fraction']:.4f}"


if __name__ == "__main__":
    main()
