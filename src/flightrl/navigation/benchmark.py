from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .scenarios import NavigationScenario, scenario_by_name


def evaluate_scenario_record(
    scenario: NavigationScenario,
    *,
    label: str,
    metrics: dict[str, float],
    checkpoint: str | None = None,
) -> dict[str, Any]:
    failures = metric_failures(scenario, metrics)
    return {
        "label": label,
        "scenario": scenario.name,
        "task": scenario.task,
        "reset_profile": scenario.reset_profile,
        "checkpoint": checkpoint,
        "drones_per_env": scenario.drones_per_env,
        "observation_source": scenario.observation_source,
        "action_interface": scenario.action_interface,
        "metrics": dict(metrics),
        "score": navigation_score(scenario, metrics),
        "passed": not failures,
        "failures": failures,
    }


def build_navigation_benchmark_report(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [evaluate_input_record(record) for record in records]
    evaluated.sort(key=lambda item: (item["scenario"], -item["score"], item["label"]))
    passed = [record for record in evaluated if record["passed"]]
    return {
        "records": evaluated,
        "best_by_scenario": best_by_scenario(evaluated),
        "summary": {
            "total_records": len(evaluated),
            "scenarios": len({record["scenario"] for record in evaluated}),
            "passed_records": len(passed),
            "blocked_records": len(evaluated) - len(passed),
            "all_passed": len(passed) == len(evaluated) and bool(evaluated),
        },
        "safety": "Simulation benchmark only; not approved for live hardware.",
    }


def evaluate_input_record(record: dict[str, Any]) -> dict[str, Any]:
    scenario = scenario_by_name(str(record["scenario"]))
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("navigation benchmark record must contain a metrics object")
    return evaluate_scenario_record(
        scenario,
        label=str(record["label"]),
        checkpoint=record.get("checkpoint"),
        metrics={str(key): float(value) for key, value in metrics.items()},
    )


def best_by_scenario(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        scenario = record["scenario"]
        if scenario not in best or record_sort_key(record) < record_sort_key(best[scenario]):
            best[scenario] = record
    return {scenario: compact_record(record) for scenario, record in best.items()}


def record_sort_key(record: dict[str, Any]) -> tuple[int, float, str]:
    return (0 if record["passed"] else 1, -float(record["score"]), str(record["label"]))


def metric_failures(scenario: NavigationScenario, metrics: dict[str, float]) -> list[str]:
    missing = [name for name in scenario.required_metrics if name not in metrics]
    if missing:
        return [f"missing_{name}" for name in missing]
    thresholds = scenario.thresholds
    checks = [
        ("mean_completed_fraction", metrics["mean_completed_fraction"], thresholds.min_completed_fraction, "lt"),
        ("mean_survival_fraction", metrics["mean_survival_fraction"], thresholds.min_survival_fraction, "lt"),
        ("mean_position_error_m", metrics["mean_position_error_m"], thresholds.max_position_error_m, "gt"),
        ("clearance_p01_m", metrics["clearance_p01_m"], thresholds.min_clearance_p01_m, "lt"),
        ("action_saturation_fraction", metrics["action_saturation_fraction"], thresholds.max_action_saturation_fraction, "gt"),
    ]
    failures = []
    for name, value, threshold, direction in checks:
        if direction == "lt" and value < threshold:
            failures.append(f"{name}_lt_{threshold:g}")
        if direction == "gt" and value > threshold:
            failures.append(f"{name}_gt_{threshold:g}")
    return failures


def navigation_score(scenario: NavigationScenario, metrics: dict[str, float]) -> float:
    missing = [name for name in scenario.required_metrics if name not in metrics]
    if missing:
        return 0.0
    thresholds = scenario.thresholds
    completion = clamp01(metrics["mean_completed_fraction"])
    survival = clamp01(metrics["mean_survival_fraction"])
    clearance = clamp01(metrics["clearance_p01_m"] / thresholds.preferred_clearance_m)
    precision = 1.0 - clamp01(metrics["mean_position_error_m"] / thresholds.max_position_error_m)
    smoothness = 1.0 - clamp01(metrics["action_saturation_fraction"] / thresholds.max_action_saturation_fraction)
    return round(0.34 * completion + 0.24 * survival + 0.18 * clearance + 0.16 * precision + 0.08 * smoothness, 6)


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": record["label"],
        "scenario": record["scenario"],
        "task": record["task"],
        "checkpoint": record["checkpoint"],
        "score": record["score"],
        "passed": record["passed"],
        "failures": record["failures"],
    }


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Navigation Benchmark Report",
        "",
        "| scenario | label | passed | score | task | reset profile | failures |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for record in report["records"]:
        failures = ", ".join(record["failures"]) or "none"
        lines.append(
            f"| {record['scenario']} | {record['label']} | {record['passed']} | {record['score']:.4f} | "
            f"{record['task']} | {record['reset_profile']} | {failures} |"
        )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
