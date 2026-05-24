from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

from flightrl.sixdof.sweep import all_success, run_commands


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = ("position_yaw_easy", "position_yaw_medium", "position_yaw_recovery", "broad")


@dataclass(frozen=True, slots=True)
class DaggerVariant:
    name: str
    beta: float
    reset_profile: str
    eval_reset_profile: str
    action_weighting: str = "none"
    iterations: int = 2
    learning_rate: float = 8e-4


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run closed-loop DAgger recovery sweeps for 6-DoF position/yaw")
    parser.add_argument("--seed-dataset", default="artifacts/datasets/sixdof_position_yaw_recovery_history1_512x192_noise008.npz")
    parser.add_argument("--initial-checkpoint", default="artifacts/checkpoints/sixdof_position_yaw_recovery_history1_512x192_noise008_h128.pt")
    parser.add_argument("--output-dir", default="artifacts/dagger/position_yaw_recovery")
    parser.add_argument("--report", default="artifacts/replay/sixdof_position_yaw_recovery_dagger_sweep.json")
    parser.add_argument("--task", default="position_yaw")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=400)
    parser.add_argument("--eval-num-envs", type=int, default=256)
    parser.add_argument("--diagnostic-steps", type=int, default=400)
    parser.add_argument("--diagnostic-num-envs", type=int, default=256)
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--native-step", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    variants = default_variants()
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    records = [variant_record(args, variant) for variant in variants]
    if args.run:
        for record in records:
            record["results"] = run_commands(record["commands"], cwd=ROOT)
            record["diagnostics"] = load_diagnostics(record["diagnostics_path"])
    report = {"run": args.run, "profiles": args.profiles, "records": records, "summary": sweep_summary(records)}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"summary={output}")
    print(f"markdown={output.with_suffix('.md')}")


def default_variants() -> list[DaggerVariant]:
    return [
        DaggerVariant("policy_states_beta000", beta=0.0, reset_profile="position_yaw_recovery", eval_reset_profile="position_yaw_recovery"),
        DaggerVariant(
            "policy_states_beta010_weighted",
            beta=0.10,
            reset_profile="position_yaw_recovery",
            eval_reset_profile="position_yaw_recovery",
            action_weighting="inverse_std",
        ),
        DaggerVariant("medium_recovery_beta005", beta=0.05, reset_profile="position_yaw_medium", eval_reset_profile="position_yaw_recovery"),
    ]


def variant_record(args: argparse.Namespace, variant: DaggerVariant) -> dict:
    base = Path(args.output_dir) / variant.name
    checkpoint = base / f"iter_{variant.iterations:02d}.pt"
    diagnostics = base / "diagnostics.json"
    return {
        "variant": asdict(variant),
        "checkpoint": str(checkpoint),
        "diagnostics_path": str(diagnostics),
        "commands": [dagger_command(args, variant, base), diagnostic_command(args, checkpoint, diagnostics)],
    }


def dagger_command(args: argparse.Namespace, variant: DaggerVariant, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_sixdof_dagger.py"),
        "--seed-dataset",
        args.seed_dataset,
        "--initial-checkpoint",
        args.initial_checkpoint,
        "--output-dir",
        str(output_dir),
        "--task",
        args.task,
        "--iterations",
        str(variant.iterations),
        "--num-envs",
        str(args.num_envs),
        "--steps",
        str(args.steps),
        "--beta",
        str(variant.beta),
        "--reset-profile",
        variant.reset_profile,
        "--eval-reset-profile",
        variant.eval_reset_profile,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-size",
        str(args.hidden_size),
        "--learning-rate",
        str(variant.learning_rate),
        "--eval-steps",
        str(args.eval_steps),
        "--eval-num-envs",
        str(args.eval_num_envs),
        "--select-by-eval",
        "--action-weighting",
        variant.action_weighting,
    ]
    if args.native_step:
        command.append("--native-step")
    return command


def diagnostic_command(args: argparse.Namespace, checkpoint: Path, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "diagnose_sixdof_policy.py"),
        "--checkpoint",
        str(checkpoint),
        "--task",
        args.task,
        "--profiles",
        *args.profiles,
        "--steps",
        str(args.diagnostic_steps),
        "--num-envs",
        str(args.diagnostic_num_envs),
        "--output",
        str(output),
    ]
    return command


def load_diagnostics(path: str) -> dict | None:
    report = Path(path)
    if not report.exists():
        return None
    data = json.loads(report.read_text())
    return {record["reset_profile"]: record["final"] for record in data.get("records", [])}


def sweep_summary(records: list[dict]) -> dict:
    return {
        "total": len(records),
        "completed": sum(1 for record in records if all_success(record.get("results"))),
        "best_recovery": best_record(records, "position_yaw_recovery"),
        "best_broad": best_record(records, "broad"),
    }


def best_record(records: list[dict], profile: str) -> dict | None:
    candidates = []
    for record in records:
        final = (record.get("diagnostics") or {}).get(profile)
        if final:
            candidates.append((diagnostic_score(final), compact_record(record, profile, final)))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def diagnostic_score(final: dict) -> tuple:
    return (
        -final["survival_fraction"],
        -final["clearance_p01_m"],
        final["position_error_mean_m"],
        final["yaw_error_mean_rad"],
    )


def compact_record(record: dict, profile: str, final: dict) -> dict:
    return {
        "name": record["variant"]["name"],
        "checkpoint": record["checkpoint"],
        "profile": profile,
        "survival_fraction": final["survival_fraction"],
        "position_error_mean_m": final["position_error_mean_m"],
        "clearance_p01_m": final["clearance_p01_m"],
        "yaw_error_mean_rad": final["yaw_error_mean_rad"],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# 6-DoF Recovery DAgger Sweep",
        "",
        "| variant | status | recovery survival | broad survival |",
        "| --- | --- | ---: | ---: |",
    ]
    for record in report["records"]:
        diagnostics = record.get("diagnostics") or {}
        lines.append(
            f"| {record['variant']['name']} | {status(record)} | "
            f"{fmt_profile(diagnostics, 'position_yaw_recovery')} | {fmt_profile(diagnostics, 'broad')} |"
        )
    summary = report.get("summary") or {}
    if summary.get("best_recovery") or summary.get("best_broad"):
        lines.extend(["", "## Best Candidates", ""])
        if summary.get("best_recovery"):
            lines.append(f"- recovery: `{summary['best_recovery']['name']}` survival `{summary['best_recovery']['survival_fraction']:.4f}`")
        if summary.get("best_broad"):
            lines.append(f"- broad: `{summary['best_broad']['name']}` survival `{summary['best_broad']['survival_fraction']:.4f}`")
    lines.extend(["", "Commands are stored in the JSON report. Checkpoints remain simulation-only until replay and hardware safety gates pass."])
    return "\n".join(lines)


def status(record: dict) -> str:
    if "results" not in record:
        return "planned"
    return "ok" if all_success(record["results"]) else "failed"


def fmt_profile(diagnostics: dict, profile: str) -> str:
    final = diagnostics.get(profile)
    return f"{final['survival_fraction']:.4f}" if final else "pending"


if __name__ == "__main__":
    main()
