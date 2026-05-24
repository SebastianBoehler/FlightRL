from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from flightrl.sixdof.sweep import all_success, run_commands


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate readiness candidates across reset profiles and build a profile matrix")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--profiles", nargs="+", default=["position_yaw_recovery", "broad"])
    parser.add_argument("--output-dir", default="artifacts/replay/sixdof_profile_gate")
    parser.add_argument("--output", default="artifacts/replay/sixdof_profile_gate.json")
    parser.add_argument("--profile-matrix-output", default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1211)
    parser.add_argument("--native-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-yaw-error-rad", type=float, default=0.35)
    parser.add_argument("--max-yaw-p95-error-rad", type=float, default=0.60)
    args = parser.parse_args()

    candidates = profile_candidates(json.loads(Path(args.matrix).read_text()))
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise SystemExit("no candidates containing position_yaw found in matrix")
    report = build_report(args, candidates)
    if args.run:
        report["results"] = run_commands(report["commands"], cwd=ROOT)
        report["profile_matrix"] = str(profile_matrix_path(args))
        report["profile_summary"] = load_profile_summary(profile_matrix_path(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"profile_gate={output}")
    print(f"markdown={output.with_suffix('.md')}")


def profile_candidates(matrix: dict) -> list[dict]:
    records = list(matrix.get("best_by_task", {}).values())
    if matrix.get("best_multitask"):
        records.append(matrix["best_multitask"])
    seen = set()
    selected = []
    for record in records:
        key = (record["label"], record["checkpoint"])
        if key in seen or "position_yaw" not in record.get("tasks", []):
            continue
        seen.add(key)
        selected.append({"label": record["label"], "checkpoint": record["checkpoint"], "tasks": record["tasks"]})
    return selected


def build_report(args: argparse.Namespace, candidates: list[dict]) -> dict:
    output_dir = Path(args.output_dir)
    suite_paths = [output_dir / f"profile_{profile}.json" for profile in args.profiles]
    commands = [suite_command(args, candidates, profile, path, idx) for idx, (profile, path) in enumerate(zip(args.profiles, suite_paths, strict=True))]
    commands.append(matrix_command(suite_paths, profile_matrix_path(args)))
    return {
        "run": args.run,
        "source_matrix": args.matrix,
        "profiles": args.profiles,
        "candidates": candidates,
        "suite_paths": [str(path) for path in suite_paths],
        "profile_matrix_output": str(profile_matrix_path(args)),
        "commands": commands,
        "safety": "Profile gate is simulation-only and does not approve live hardware.",
    }


def suite_command(args: argparse.Namespace, candidates: list[dict], profile: str, output: Path, index: int) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_sixdof_suite.py"),
        "--steps",
        str(args.steps),
        "--num-envs",
        str(args.num_envs),
        "--seed",
        str(args.seed + index),
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
    for candidate in candidates:
        command.extend(["--candidate", candidate["label"], candidate["checkpoint"], ",".join(candidate["tasks"])])
    return command


def matrix_command(suite_paths: list[Path], output: Path) -> list[str]:
    command = [sys.executable, str(ROOT / "scripts" / "build_sixdof_profile_matrix.py")]
    for path in suite_paths:
        command.extend(["--suite", str(path)])
    command.extend(["--output", str(output)])
    return command


def profile_matrix_path(args: argparse.Namespace) -> Path:
    if args.profile_matrix_output:
        return Path(args.profile_matrix_output)
    return Path(args.output_dir) / "profile_matrix.json"


def load_profile_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    report = json.loads(path.read_text())
    return [
        {
            "label": record["label"],
            "tasks": record["tasks"],
            "passed_all_profiles": record["passed_all_profiles"],
            "worst_survival_fraction": record["worst_survival_fraction"],
            "worst_completed_fraction": record["worst_completed_fraction"],
            "worst_position_error_m": record["worst_position_error_m"],
            "worst_yaw_error_rad": record.get("worst_yaw_error_rad"),
            "worst_clearance_p01_m": record["worst_clearance_p01_m"],
        }
        for record in report.get("records", [])
    ]


def render_markdown(report: dict) -> str:
    lines = [
        "# 6-DoF Profile Gate",
        "",
        f"Profiles: `{', '.join(report['profiles'])}`",
        "",
        "| candidate | tasks |",
        "| --- | --- |",
    ]
    for candidate in report["candidates"]:
        lines.append(f"| {candidate['label']} | {', '.join(candidate['tasks'])} |")
    status = "planned"
    if report.get("results"):
        status = "ok" if all_success(report["results"]) else "failed"
    lines.extend(["", f"Status: `{status}`", f"Profile matrix: `{report['profile_matrix_output']}`"])
    if report.get("profile_summary"):
        lines.extend(
            [
                "",
                "| candidate | all passed | worst survival | worst completed | worst pos err m | worst yaw rad | worst clearance m |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for record in report["profile_summary"]:
            lines.append(
                f"| {record['label']} | {record['passed_all_profiles']} | {fmt(record['worst_survival_fraction'])} | "
                f"{fmt(record['worst_completed_fraction'])} | {fmt(record['worst_position_error_m'])} | "
                f"{fmt(record.get('worst_yaw_error_rad'))} | {fmt(record['worst_clearance_p01_m'])} |"
            )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
