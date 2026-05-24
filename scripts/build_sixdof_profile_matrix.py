from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate 6-DoF validation suites by candidate and reset profile")
    parser.add_argument("--suite", action="append", required=True, help="Validation suite JSON artifact. Repeatable.")
    parser.add_argument("--output", default="artifacts/replay/sixdof_profile_matrix.json")
    args = parser.parse_args()

    report = build_profile_matrix([Path(path) for path in args.suite])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"profile_matrix={output}")
    print(f"markdown={output.with_suffix('.md')}")


def build_profile_matrix(suites: list[Path]) -> dict:
    grouped: dict[tuple, dict] = {}
    task_grouped: dict[tuple, dict] = {}
    profiles: list[str] = []
    for suite_path in suites:
        suite = json.loads(suite_path.read_text())
        profile = suite.get("reset_profile", "broad")
        if profile not in profiles:
            profiles.append(profile)
        for record in suite["records"]:
            if record["controller"] == "teacher":
                continue
            key = (record["label"], record["checkpoint"], tuple(record["tasks"]))
            candidate = grouped.setdefault(
                key,
                {"label": record["label"], "controller": record["controller"], "checkpoint": record["checkpoint"], "tasks": record["tasks"], "profiles": {}},
            )
            candidate["profiles"][profile] = compact_profile(record, suite_path)
            add_task_profiles(task_grouped, record, profile, suite_path)
    records = [summarize_candidate(candidate, profiles) for candidate in grouped.values()]
    task_records = [summarize_task(task, profiles) for task in task_grouped.values()]
    return {
        "profiles": profiles,
        "records": sorted(records, key=score),
        "task_records": sorted(task_records, key=task_score),
        "best_by_task": best_by_task(records),
        "safety": "Profile matrix is simulation-only and does not approve live hardware.",
    }


def compact_profile(record: dict, suite_path: Path) -> dict:
    metrics = record["metrics"]
    gate = record["gate"]
    return {
        "suite": str(suite_path),
        "passed": gate["passed"],
        "failures": gate["failures"],
        "mean_completed_fraction": metrics["mean_completed_fraction"],
        "mean_survival_fraction": metrics.get("mean_survival_fraction", metrics["mean_completed_fraction"]),
        "mean_position_error_m": metrics["mean_position_error_m"],
        "mean_yaw_error_rad": metrics.get("mean_yaw_error_rad"),
        "yaw_error_p95_rad": metrics.get("yaw_error_p95_rad"),
        "clearance_p01_m": metrics.get("clearance_p01_m", metrics["min_clearance_m"]),
    }


def add_task_profiles(grouped: dict[tuple, dict], record: dict, profile: str, suite_path: Path) -> None:
    for task, metrics in record["metrics"].get("per_task", {}).items():
        key = (record["label"], record["checkpoint"], task)
        item = grouped.setdefault(
            key,
            {"label": record["label"], "checkpoint": record["checkpoint"], "task": task, "profiles": {}},
        )
        item["profiles"][profile] = compact_task_profile(metrics, record.get("per_task_gate", {}).get(task), suite_path)


def compact_task_profile(metrics: dict, gate: dict | None, suite_path: Path) -> dict:
    return {
        "suite": str(suite_path),
        "passed": bool((gate or {}).get("passed", False)),
        "failures": (gate or {}).get("failures", []),
        "completed_fraction": metrics["completed_fraction"],
        "survival_fraction": metrics.get("survival_fraction", metrics["completed_fraction"]),
        "mean_position_error_m": metrics["mean_position_error_m"],
        "mean_yaw_error_rad": metrics.get("mean_yaw_error_rad"),
        "yaw_error_p95_rad": metrics.get("yaw_error_p95_rad"),
        "clearance_p01_m": metrics.get("clearance_p01_m", metrics["min_clearance_m"]),
    }


def summarize_candidate(candidate: dict, required_profiles: list[str]) -> dict:
    profiles = candidate["profiles"]
    missing = [profile for profile in required_profiles if profile not in profiles]
    values = list(profiles.values())
    failures = {profile: data["failures"] for profile, data in profiles.items() if not data["passed"]}
    passed_all = not missing and all(data["passed"] for data in values)
    return {
        **candidate,
        "missing_profiles": missing,
        "passed_all_profiles": passed_all,
        "failures_by_profile": failures,
        "worst_completed_fraction": min_value(values, "mean_completed_fraction"),
        "worst_survival_fraction": min_value(values, "mean_survival_fraction"),
        "worst_clearance_p01_m": min_value(values, "clearance_p01_m"),
        "worst_position_error_m": max_value(values, "mean_position_error_m"),
        "worst_yaw_error_rad": max_optional(values, "mean_yaw_error_rad"),
        "worst_yaw_p95_rad": max_optional(values, "yaw_error_p95_rad"),
    }


def summarize_task(record: dict, required_profiles: list[str]) -> dict:
    profiles = record["profiles"]
    values = list(profiles.values())
    missing = [profile for profile in required_profiles if profile not in profiles]
    failures = {profile: data["failures"] for profile, data in profiles.items() if not data["passed"]}
    return {
        **record,
        "missing_profiles": missing,
        "passed_all_profiles": not missing and all(data["passed"] for data in values),
        "failures_by_profile": failures,
        "worst_completed_fraction": min_value(values, "completed_fraction"),
        "worst_survival_fraction": min_value(values, "survival_fraction"),
        "worst_clearance_p01_m": min_value(values, "clearance_p01_m"),
        "worst_position_error_m": max_value(values, "mean_position_error_m"),
        "worst_yaw_error_rad": max_optional(values, "mean_yaw_error_rad"),
        "worst_yaw_p95_rad": max_optional(values, "yaw_error_p95_rad"),
    }


def min_value(values: list[dict], key: str) -> float | None:
    return min((float(value[key]) for value in values if value.get(key) is not None), default=None)


def max_value(values: list[dict], key: str) -> float | None:
    return max((float(value[key]) for value in values if value.get(key) is not None), default=None)


def max_optional(values: list[dict], key: str) -> float | None:
    return max((float(value[key]) for value in values if value.get(key) is not None), default=None)


def best_by_task(records: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for record in records:
        if len(record["tasks"]) != 1:
            continue
        task = record["tasks"][0]
        if task not in best or score(record) < score(best[task]):
            best[task] = record
    return {task: compact_candidate(record) for task, record in best.items()}


def score(record: dict) -> tuple:
    return (
        0 if record["passed_all_profiles"] else 1,
        len(record["missing_profiles"]),
        -safe(record["worst_survival_fraction"]),
        -safe(record["worst_completed_fraction"]),
        safe(record["worst_position_error_m"], high=True),
        safe(record["worst_yaw_error_rad"], high=True),
        -safe(record["worst_clearance_p01_m"]),
    )


def task_score(record: dict) -> tuple:
    return (
        0 if not record["passed_all_profiles"] else 1,
        safe(record["worst_completed_fraction"]),
        safe(record["worst_survival_fraction"]),
        -safe(record["worst_position_error_m"], high=True),
        -safe(record["worst_yaw_p95_rad"], high=True),
        -safe(record["worst_yaw_error_rad"], high=True),
        safe(record["worst_clearance_p01_m"]),
    )


def safe(value: float | None, *, high: bool = False) -> float:
    if value is None:
        return 999_999.0 if high else 0.0
    return float(value)


def compact_candidate(record: dict) -> dict:
    return {
        "label": record["label"],
        "controller": record.get("controller", "checkpoint"),
        "checkpoint": record["checkpoint"],
        "tasks": record["tasks"],
        "passed_all_profiles": record["passed_all_profiles"],
        "missing_profiles": record["missing_profiles"],
        "failures_by_profile": record["failures_by_profile"],
        "worst_survival_fraction": record["worst_survival_fraction"],
        "worst_completed_fraction": record["worst_completed_fraction"],
        "worst_position_error_m": record["worst_position_error_m"],
        "worst_clearance_p01_m": record["worst_clearance_p01_m"],
        "worst_yaw_error_rad": record["worst_yaw_error_rad"],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# 6-DoF Profile Matrix",
        "",
        f"Profiles: `{', '.join(report['profiles'])}`",
        "",
        "| label | controller | tasks | all passed | missing | worst survival | worst completed | worst pos err m | worst yaw rad | worst clearance m |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in report["records"]:
        lines.append(
            f"| {record['label']} | {record.get('controller', 'checkpoint')} | {', '.join(record['tasks'])} | {record['passed_all_profiles']} | "
            f"{', '.join(record['missing_profiles']) or 'none'} | {fmt(record['worst_survival_fraction'])} | "
            f"{fmt(record['worst_completed_fraction'])} | {fmt(record['worst_position_error_m'])} | "
            f"{fmt(record['worst_yaw_error_rad'])} | {fmt(record['worst_clearance_p01_m'])} |"
        )
    if report.get("task_records"):
        lines.extend(
            [
                "",
                "## Per-Task Blockers",
                "",
                "| label | task | all passed | worst completed | worst pos err m | worst yaw rad | yaw p95 rad | worst clearance m |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for record in report["task_records"][:12]:
            lines.append(
                f"| {record['label']} | {record['task']} | {record['passed_all_profiles']} | "
                f"{fmt(record['worst_completed_fraction'])} | {fmt(record['worst_position_error_m'])} | "
                f"{fmt(record['worst_yaw_error_rad'])} | {fmt(record['worst_yaw_p95_rad'])} | "
                f"{fmt(record['worst_clearance_p01_m'])} |"
            )
    if report["best_by_task"]:
        lines.extend(["", "## Best By Task", ""])
        for task, record in report["best_by_task"].items():
            lines.append(
                f"- `{task}`: `{record['label']}` all_passed=`{record['passed_all_profiles']}` "
                f"worst_survival=`{fmt(record['worst_survival_fraction'])}`"
            )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
