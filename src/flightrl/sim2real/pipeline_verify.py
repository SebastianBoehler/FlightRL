from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightrl.sim2real.provenance import path_provenance, path_provenance_failure


def verify_pipeline(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    records = [record for name, value in report.get("inputs", {}).items() for record in verify_input(name, value)]
    failures = [record["failure"] for record in records if record["failure"]]
    return {
        "pipeline": str(path),
        "passed": not failures,
        "failures": failures,
        "records": records,
        "safety": "Verification checks offline pipeline input freshness only; it does not approve live hardware deployment.",
    }


def verify_input(name: str, value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and "path" in value:
        return [verify_path(name, value)]
    if isinstance(value, list):
        records: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            records.extend(verify_input(f"{name}[{index}]", item))
        return records
    return []


def verify_path(name: str, expected: dict[str, Any]) -> dict[str, Any]:
    path_text = expected.get("path")
    current = path_provenance(Path(path_text)) if path_text else {"path": path_text, "exists": False}
    failure = path_provenance_failure(expected, current)
    return {
        "name": name,
        "path": path_text,
        "expected": expected,
        "current": current,
        "passed": failure is None,
        "failure": failure,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sim-To-Real Pipeline Verification",
        "",
        f"- Passed: `{report['passed']}`",
        f"- Pipeline: `{report['pipeline']}`",
        "",
        "| input | passed | failure | path |",
        "| --- | ---: | --- | --- |",
    ]
    for record in report["records"]:
        lines.append(f"| {record['name']} | {record['passed']} | {record['failure'] or 'none'} | `{record['path']}` |")
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
