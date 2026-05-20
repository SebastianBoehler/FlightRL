from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a ranked matrix from 6-DoF validation suite artifacts")
    parser.add_argument("--suite", action="append", required=True, help="Validation suite JSON artifact. Repeatable.")
    parser.add_argument("--parity", action="append", default=[], help="Optional LABEL=parity.json edge export report. Repeatable.")
    parser.add_argument("--output", default="artifacts/replay/sixdof_candidate_matrix.json")
    parser.add_argument("--max-parity-error", type=float, default=1e-5)
    args = parser.parse_args()

    parity = load_parity(args.parity)
    records = [record for suite in args.suite for record in checkpoint_records(Path(suite), parity, args.max_parity_error)]
    report = {"records": records, "best_by_task": best_by_task(records), "safety": "Simulation ranking only; not approved for live hardware."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"matrix={output}")
    print(f"markdown={output.with_suffix('.md')}")


def load_parity(items: list[str]) -> dict[str, dict]:
    reports = {}
    for item in items:
        if "=" not in item:
            raise SystemExit("--parity must be LABEL=path")
        label, path = item.split("=", 1)
        reports[label] = json.loads(Path(path).read_text())
    return reports


def checkpoint_records(suite_path: Path, parity: dict[str, dict], max_parity_error: float) -> list[dict]:
    suite = json.loads(suite_path.read_text())
    records = []
    for record in suite["records"]:
        if record["controller"] != "checkpoint":
            continue
        checkpoint = record["checkpoint"]
        gate = record["gate"]
        metrics = record["metrics"]
        label = record["label"]
        parity_report = parity.get(label)
        records.append(
            {
                "label": label,
                "suite": str(suite_path),
                "checkpoint": checkpoint,
                "tasks": record["tasks"],
                "passed": gate["passed"],
                "failures": gate["failures"],
                "mean_completed_fraction": metrics["mean_completed_fraction"],
                "mean_survival_fraction": metrics.get("mean_survival_fraction", metrics["mean_completed_fraction"]),
                "mean_position_error_m": metrics["mean_position_error_m"],
                "clearance_p01_m": metrics.get("clearance_p01_m", metrics["min_clearance_m"]),
                "teacher_action_l2_mean": metrics.get("teacher_action_l2_mean"),
                "action_saturation_fraction": metrics.get("action_saturation_fraction"),
                "checkpoint_meta": checkpoint_meta(checkpoint),
                "edge_parity": compact_parity(parity_report, max_parity_error),
            }
        )
    return records


def checkpoint_meta(path: str | None) -> dict:
    if not path or not Path(path).exists():
        return {}
    checkpoint = torch.load(path, map_location="cpu")
    return {
        "trainer": checkpoint.get("trainer", "unknown"),
        "hidden_size": checkpoint.get("hidden_size"),
        "observation_dim": checkpoint.get("observation_dim", 28),
        "observation_mode": checkpoint.get("observation_mode", "base"),
        "task_conditioned": checkpoint.get("task_conditioned", False),
    }


def compact_parity(report: dict | None, max_error: float) -> dict:
    if not report:
        return {"present": False, "passed": False}
    error = float(report["parity"]["max_abs_error"])
    return {
        "present": True,
        "passed": error <= max_error,
        "max_abs_error": error,
        "model": report.get("model"),
        "observation_mode": report.get("observation", {}).get("mode", "base"),
    }


def best_by_task(records: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for record in records:
        if len(record["tasks"]) != 1:
            continue
        task = record["tasks"][0]
        if task not in best or score(record) < score(best[task]):
            best[task] = record
    return {task: compact_record(record) for task, record in best.items()}


def score(record: dict) -> tuple:
    return (
        0 if record["passed"] else 1,
        -record["mean_completed_fraction"],
        -record["mean_survival_fraction"],
        record["mean_position_error_m"],
        -record["clearance_p01_m"],
        0 if record["edge_parity"]["present"] else 1,
    )


def compact_record(record: dict) -> dict:
    return {key: record[key] for key in ("label", "checkpoint", "passed", "failures", "mean_completed_fraction", "mean_position_error_m", "clearance_p01_m", "edge_parity")}


def render_markdown(report: dict) -> str:
    lines = ["# 6-DoF Candidate Matrix", "", "| label | tasks | passed | edge | completed | survival | pos err m | clearance p01 m | obs mode |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for record in sorted(report["records"], key=score):
        edge = "yes" if record["edge_parity"]["present"] else "no"
        meta = record["checkpoint_meta"]
        lines.append(
            f"| {record['label']} | {', '.join(record['tasks'])} | {record['passed']} | {edge} | "
            f"{record['mean_completed_fraction']:.4f} | {record['mean_survival_fraction']:.4f} | "
            f"{record['mean_position_error_m']:.4f} | {record['clearance_p01_m']:.4f} | {meta.get('observation_mode', 'unknown')} |"
        )
    if report["best_by_task"]:
        lines.extend(["", "## Best By Task", ""])
        for task, record in report["best_by_task"].items():
            lines.append(f"- `{task}`: `{record['label']}` passed=`{record['passed']}` completed=`{record['mean_completed_fraction']:.4f}`")
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
