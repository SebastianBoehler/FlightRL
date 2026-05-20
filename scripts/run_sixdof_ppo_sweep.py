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
class PpoVariant:
    name: str
    learning_rate: float
    action_std: float
    imitation_coef: float
    reference_coef: float
    updates: int = 16
    update_epochs: int = 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run 6-DoF position/yaw PPO tuning sweeps")
    parser.add_argument("--init-checkpoint", default="artifacts/curriculum/position_yaw/easy_medium_h128/checkpoint.pt")
    parser.add_argument("--output-dir", default="artifacts/ppo/position_yaw")
    parser.add_argument("--report", default="artifacts/replay/sixdof_position_yaw_ppo_sweep.json")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--native-step", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    variants = default_variants()
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    records = [variant_record(args, variant) for variant in variants]
    if args.run:
        for record in records:
            record["results"] = run_commands(record["commands"])
            record["gates"] = load_gate_summaries(record)
    report = {"run": args.run, "records": records, "summary": sweep_summary(records)}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"summary={output}")
    print(f"markdown={output.with_suffix('.md')}")


def default_variants() -> list[PpoVariant]:
    return [
        PpoVariant("ref1_std006_lr5e5", 5e-5, 0.06, 0.05, 1.0),
        PpoVariant("ref2_std006_lr5e5", 5e-5, 0.06, 0.05, 2.0),
        PpoVariant("ref1_std004_lr3e5", 3e-5, 0.04, 0.05, 1.0),
        PpoVariant("teacher025_ref1_std006", 5e-5, 0.06, 0.25, 1.0),
    ]


def variant_record(args: argparse.Namespace, variant: PpoVariant) -> dict:
    base = Path(args.output_dir) / variant.name
    checkpoint = base / "checkpoint.pt"
    medium_gate = base / "medium_gate.json"
    broad_gate = base / "broad_gate.json"
    commands = [
        train_command(args.init_checkpoint, checkpoint, variant, args.native_step),
        eval_command(checkpoint, medium_gate, "position_yaw_medium", 400, args.native_step),
        eval_command(checkpoint, broad_gate, "broad", 800, args.native_step),
    ]
    return {
        "variant": asdict(variant),
        "checkpoint": str(checkpoint),
        "medium_gate": str(medium_gate),
        "broad_gate": str(broad_gate),
        "commands": commands,
    }


def train_command(init_checkpoint: str, checkpoint: Path, variant: PpoVariant, native_step: bool) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_sixdof_ppo.py"),
        "--init-checkpoint",
        init_checkpoint,
        "--checkpoint",
        str(checkpoint),
        "--updates",
        str(variant.updates),
        "--num-envs",
        "512",
        "--horizon",
        "64",
        "--hidden-size",
        "128",
        "--learning-rate",
        str(variant.learning_rate),
        "--update-epochs",
        str(variant.update_epochs),
        "--minibatch-size",
        "8192",
        "--action-std",
        str(variant.action_std),
        "--imitation-coef",
        str(variant.imitation_coef),
        "--reference-coef",
        str(variant.reference_coef),
        "--reset-profile",
        "position_yaw_medium",
        "--eval-reset-profile",
        "position_yaw_medium",
        "--eval-steps",
        "400",
    ]
    if native_step:
        command.append("--native-step")
    return command


def eval_command(checkpoint: Path, output: Path, reset_profile: str, steps: int, native_step: bool) -> list[str]:
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
    ]
    if native_step:
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
        "clearance_p01_m": metrics["clearance_p01_m"],
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
        if gate:
            candidates.append((gate_score(gate), compact_record(record, gate)))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def gate_score(gate: dict) -> tuple:
    return (0 if gate["passed"] else 1, -gate["mean_completed_fraction"], -gate["mean_survival_fraction"], gate["mean_position_error_m"])


def compact_record(record: dict, gate: dict) -> dict:
    return {
        "name": record["variant"]["name"],
        "checkpoint": record["checkpoint"],
        "passed": gate["passed"],
        "failures": gate["failures"],
        "mean_completed_fraction": gate["mean_completed_fraction"],
        "mean_survival_fraction": gate["mean_survival_fraction"],
        "mean_position_error_m": gate["mean_position_error_m"],
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
