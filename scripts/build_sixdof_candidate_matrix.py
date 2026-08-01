from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flightrl.sixdof import require_current_checkpoint
from flightrl.evidence_scope import (
    DESKTOP_CPU_SCOPE,
    DESKTOP_DEVELOPMENT_SCOPE,
    require_existing_file_identity,
    require_file_identity,
)
from flightrl.sixdof.candidate_evidence import compact_latency, compact_parity, validate_suite_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a ranked matrix from 6-DoF validation suite artifacts")
    parser.add_argument("--suite", action="append", required=True, help="Validation suite JSON artifact. Repeatable.")
    parser.add_argument("--desktop-parity", action="append", default=[], help="Optional LABEL=parity.json desktop CPU export report. Repeatable.")
    parser.add_argument(
        "--desktop-latency",
        action="append",
        default=[],
        help="Optional LABEL=latency.json desktop benchmark report. Repeatable.",
    )
    parser.add_argument("--output", default="artifacts/replay/sixdof_candidate_matrix.json")
    parser.add_argument("--max-parity-error", type=float, default=1e-5)
    args = parser.parse_args()

    parity = load_desktop_parity(args.desktop_parity)
    latency = load_labeled_reports(args.desktop_latency, "--desktop-latency")
    records = [record for suite in args.suite for record in checkpoint_records(Path(suite), parity, latency, args.max_parity_error)]
    report = {
        "evidence_scope": DESKTOP_DEVELOPMENT_SCOPE,
        "deployment_authority": False,
        "records": records,
        "best_by_task": best_by_task(records),
        "best_multitask": best_multitask(records),
        "safety": "Simulation ranking with desktop CPU evidence only; not AI Deck deployment readiness or live-hardware authority.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"matrix={output}")
    print(f"markdown={output.with_suffix('.md')}")


def load_labeled_reports(items: list[str], flag: str) -> dict[str, dict]:
    reports = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"{flag} must be LABEL=path")
        label, path = item.split("=", 1)
        if not label or label in reports:
            raise SystemExit(f"{flag} labels must be nonempty and unique")
        report = json.loads(Path(path).read_text())
        if not isinstance(report, dict):
            raise SystemExit(f"{flag} reports must be JSON objects")
        reports[label] = report
    return reports


def load_desktop_parity(items: list[str]) -> dict[str, dict]:
    return load_labeled_reports(items, "--desktop-parity")


def checkpoint_records(suite_path: Path, parity: dict[str, dict], latency: dict[str, dict], max_parity_error: float) -> list[dict]:
    suite = json.loads(suite_path.read_text())
    if not isinstance(suite, dict) or not isinstance(suite.get("records"), list):
        raise ValueError(f"validation suite records are missing or invalid: {suite_path}")
    records = []
    for record in suite["records"]:
        validate_suite_record(record)
        if record["controller"] == "teacher":
            continue
        checkpoint = record["checkpoint"]
        gate = record["gate"]
        metrics = record["metrics"]
        label = record["label"]
        parity_report = parity[label] if label in parity else None
        latency_report = latency[label] if label in latency else None
        validate_desktop_evidence(checkpoint, parity_report, latency_report)
        records.append(
            {
                "label": label,
                "controller": record["controller"],
                "suite": str(suite_path),
                "checkpoint": checkpoint,
                "tasks": record["tasks"],
                "passed": gate["passed"],
                "failures": gate["failures"],
                "mean_completed_fraction": metrics["mean_completed_fraction"],
                "mean_survival_fraction": metrics.get("mean_survival_fraction", metrics["mean_completed_fraction"]),
                "mean_position_error_m": metrics["mean_position_error_m"],
                "mean_yaw_error_rad": metrics.get("mean_yaw_error_rad"),
                "yaw_error_p95_rad": metrics.get("yaw_error_p95_rad"),
                "clearance_p01_m": metrics.get("clearance_p01_m", metrics["min_clearance_m"]),
                "teacher_action_l2_mean": metrics.get("teacher_action_l2_mean"),
                "action_saturation_fraction": metrics.get("action_saturation_fraction"),
                "per_task_gate": record.get("per_task_gate", {}),
                "checkpoint_meta": checkpoint_meta(checkpoint),
                "desktop_parity": compact_parity(parity_report, max_parity_error),
                "desktop_latency": compact_latency(latency_report),
            }
        )
    return records


def checkpoint_meta(path: str | None) -> dict:
    if not path or not Path(path).exists():
        return {}
    checkpoint = torch.load(path, map_location="cpu")
    metadata = require_current_checkpoint(checkpoint)
    return {
        "trainer": checkpoint.get("trainer", "unknown"),
        "hidden_size": metadata.hidden_size,
        "observation_dim": metadata.observation_dim,
        "observation_mode": metadata.observation_mode,
        "task_conditioned": len(metadata.tasks) > 1,
        "controller": metadata.controller,
        "residual_scale": metadata.residual_scale,
    }


def validate_desktop_evidence(
    checkpoint: str,
    parity: dict | None,
    latency: dict | None,
) -> None:
    if parity is not None:
        require_desktop_report(
            parity,
            schema="flightrl.sixdof.desktop_export.v1",
            checkpoint=checkpoint,
            label="desktop parity",
        )
        require_existing_file_identity(
            parity.get("model"),
            label="desktop parity model",
        )
    if latency is not None:
        require_desktop_report(
            latency,
            schema="flightrl.sixdof.desktop_latency.v1",
            checkpoint=checkpoint,
            label="desktop latency",
        )
        model = latency.get("torchscript")
        if "torchscript_result" in latency:
            if model is None:
                raise ValueError("desktop latency TorchScript identity is missing")
            require_existing_file_identity(
                model,
                label="desktop latency model",
            )
            if parity is not None and model != parity.get("model"):
                raise ValueError(
                    "desktop parity and latency reference different models"
                )


def require_desktop_report(
    report: dict,
    *,
    schema: str,
    checkpoint: str,
    label: str,
) -> None:
    if (
        report.get("schema") != schema
        or report.get("evidence_scope") != DESKTOP_CPU_SCOPE
        or report.get("deployment_authority") is not False
    ):
        raise ValueError(f"{label} scope or schema is invalid")
    require_file_identity(
        report.get("checkpoint"),
        checkpoint,
        label=f"{label} checkpoint",
    )


def best_by_task(records: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for record in records:
        if len(record["tasks"]) != 1:
            continue
        task = record["tasks"][0]
        if task not in best or score(record) < score(best[task]):
            best[task] = record
    return {task: compact_record(record) for task, record in best.items()}


def best_multitask(records: list[dict]) -> dict | None:
    candidates = [record for record in records if len(record["tasks"]) > 1]
    return compact_record(min(candidates, key=score)) if candidates else None


def score(record: dict) -> tuple:
    return (
        0 if record["passed"] else 1,
        -record["mean_completed_fraction"],
        -record["mean_survival_fraction"],
        record["mean_position_error_m"],
        yaw_score(record),
        -record["clearance_p01_m"],
        0 if record["desktop_parity"].get("passed", False) else 1,
        latency_score(record),
    )


def yaw_score(record: dict) -> float:
    if record.get("mean_yaw_error_rad") is None:
        return 0.0 if "position_yaw" not in record["tasks"] else 999.0
    return float(record["mean_yaw_error_rad"])


def latency_score(record: dict) -> float:
    latency = record.get("desktop_latency", {}).get("per_sample_us")
    return float(latency) if latency is not None else 999_999.0


def compact_record(record: dict) -> dict:
    compact = {
        key: record[key]
        for key in ("label", "checkpoint", "passed", "failures", "mean_completed_fraction", "mean_position_error_m", "clearance_p01_m", "desktop_parity")
    }
    compact["tasks"] = record["tasks"]
    compact["controller"] = record["controller"]
    compact["checkpoint_meta"] = record.get("checkpoint_meta", {})
    compact["mean_yaw_error_rad"] = record.get("mean_yaw_error_rad")
    compact["yaw_error_p95_rad"] = record.get("yaw_error_p95_rad")
    compact["teacher_action_l2_mean"] = record.get("teacher_action_l2_mean")
    compact["per_task_gate"] = record.get("per_task_gate", {})
    compact["desktop_latency"] = record.get("desktop_latency", {"present": False})
    return compact


def render_markdown(report: dict) -> str:
    lines = ["# 6-DoF Candidate Matrix", "", "| label | controller | tasks | passed | desktop parity | desktop latency us | completed | survival | pos err m | yaw err rad | yaw p95 rad | teacher L2 | clearance p01 m | obs mode |", "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for record in sorted(report["records"], key=score):
        parity = parity_text(record["desktop_parity"])
        latency = record["desktop_latency"].get("per_sample_us")
        latency_text = f"{latency:.3f}" if latency is not None else "n/a"
        meta = record["checkpoint_meta"]
        lines.append(
            f"| {record['label']} | {record['controller']} | {', '.join(record['tasks'])} | {record['passed']} | {parity} | {latency_text} | "
            f"{record['mean_completed_fraction']:.4f} | {record['mean_survival_fraction']:.4f} | "
            f"{record['mean_position_error_m']:.4f} | {fmt(record.get('mean_yaw_error_rad'))} | "
            f"{fmt(record.get('yaw_error_p95_rad'))} | {fmt(record.get('teacher_action_l2_mean'))} | "
            f"{record['clearance_p01_m']:.4f} | {meta.get('observation_mode', 'unknown')} |"
        )
    if report["best_by_task"]:
        lines.extend(["", "## Best By Task", ""])
        for task, record in report["best_by_task"].items():
            lines.append(f"- `{task}`: `{record['label']}` passed=`{record['passed']}` completed=`{record['mean_completed_fraction']:.4f}`")
    if report.get("best_multitask"):
        record = report["best_multitask"]
        lines.extend(["", "## Best Multitask", "", f"- `{record['label']}` tasks=`{', '.join(record['tasks'])}` passed=`{record['passed']}` completed=`{record['mean_completed_fraction']:.4f}`"])
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def parity_text(parity: dict) -> str:
    if parity.get("passed"):
        return "pass"
    return "fail" if parity.get("present") else "missing"


def fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


if __name__ == "__main__":
    main()
