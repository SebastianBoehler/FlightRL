from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_transfer_gate(
    *,
    audit: Path,
    profile: Path,
    config_export: Path,
    deployment_readiness: Path,
    sim_readiness: Path | None = None,
    room_report: Path | None = None,
    live_safety: Path | None = None,
) -> dict[str, Any]:
    audit_data = read_json(audit)
    profile_data = read_json(profile)
    export_data = read_json(config_export)
    deployment_data = read_json(deployment_readiness)
    sim_data = read_json(sim_readiness) if sim_readiness else {}
    checks = [
        check("audit", audit, bool(audit_data.get("transfer_ready")), audit_data.get("blocking_items", [])),
        check("profile", profile, bool(profile_data.get("summary", {}).get("profile_ready")), profile_data.get("summary", {}).get("failures", [])),
        check("config_export", config_export, bool(export_data.get("exported")), export_data.get("failures", [])),
        readiness_check("deployment_readiness", deployment_readiness, deployment_data),
    ]
    if sim_readiness:
        checks.append(readiness_check("sim_readiness", sim_readiness, sim_data))
    if room_report:
        checks.append(room_check(room_report, read_json(room_report)))
    if live_safety:
        checks.append(live_safety_check(live_safety, read_json(live_safety)))
    failures = [failure for item in checks for failure in item["failures"]]
    return {
        "transfer_approved": not failures,
        "checks": checks,
        "summary": {"passed": sum(1 for item in checks if item["passed"]), "total": len(checks), "failures": failures},
        "safety": "Transfer gate is evidence only; live hardware still requires manual preflight and supervision.",
    }


def check(name: str, path: Path, passed: bool, failures: list[str]) -> dict[str, Any]:
    return {"name": name, "path": str(path), "passed": passed, "failures": [] if passed else failures or [f"{name}_failed"]}


def readiness_check(name: str, path: Path, report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    blocked = int(summary.get("blocked", 1) or 0)
    total = int(summary.get("total", 0) or 0)
    ready = int(summary.get("ready", 0) or 0)
    passed = total > 0 and blocked == 0 and ready == total
    failures = [] if passed else readiness_failures(report)
    return {"name": name, "path": str(path), "passed": passed, "failures": failures, "ready": ready, "total": total, "blocked": blocked}


def readiness_failures(report: dict[str, Any]) -> list[str]:
    failures = []
    for record in report.get("records", []):
        if not record.get("ready", False):
            task = record.get("task", "unknown")
            record_failures = record.get("failures", []) or ["not_ready"]
            failures.extend(f"{task}:{failure}" for failure in record_failures)
    return failures or ["readiness_blocked"]


def room_check(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    ready = bool(summary.get("mapping_ready", False))
    failures = [] if ready else summary.get("failures", []) or ["room_map_not_ready"]
    estimate = report.get("room_estimate", {})
    return {
        "name": "room_map",
        "path": str(path),
        "passed": ready,
        "failures": failures,
        "point_count": summary.get("point_count"),
        "width_m": estimate.get("width_m"),
        "depth_m": estimate.get("depth_m"),
        "height_m": estimate.get("height_m"),
    }


def live_safety_check(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    passed = bool(summary.get("passed", False))
    failures = [] if passed else summary.get("failures", []) or ["live_hardware_safety_failed"]
    return {
        "name": "live_hardware_safety",
        "path": str(path),
        "passed": passed,
        "failures": failures,
        "hardware_scripts": summary.get("hardware_scripts"),
        "learned_checkpoint_hardware_scripts": summary.get("learned_checkpoint_hardware_scripts"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sim-To-Real Transfer Gate",
        "",
        f"- Transfer approved: `{report['transfer_approved']}`",
        f"- Passed checks: `{report['summary']['passed']}/{report['summary']['total']}`",
        "",
        "| check | passed | failures | path |",
        "| --- | ---: | --- | --- |",
    ]
    for item in report["checks"]:
        lines.append(f"| {item['name']} | {item['passed']} | {', '.join(item['failures']) or 'none'} | `{item['path']}` |")
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def read_json(path: Path | None) -> dict[str, Any]:
    return json.loads(path.read_text()) if path else {}
