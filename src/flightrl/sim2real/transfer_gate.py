from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightrl.evidence_values import exact_nonnegative_int, failure_strings, finite_number
from flightrl.evidence_scope import DESKTOP_DEVELOPMENT_SCOPE
from flightrl.sim2real.deployment_evidence import deployment_contract_failures


def build_transfer_gate(
    *,
    audit: Path,
    profile: Path,
    config_export: Path,
    deployment_readiness: Path,
    sim_readiness: Path | None = None,
    room_report: Path | None = None,
    live_safety: Path | None = None,
    hardware_blockers: Path | None = None,
) -> dict[str, Any]:
    audit_data = read_json(audit)
    profile_data = read_json(profile)
    export_data = read_json(config_export)
    deployment_data = read_json(deployment_readiness)
    sim_data = read_json(sim_readiness) if sim_readiness else {}
    profile_summary = report_summary(profile_data)
    checks = [
        check("audit", audit, audit_data.get("transfer_ready"), audit_data.get("blocking_items", [])),
        check("profile", profile, profile_summary.get("profile_ready"), profile_summary.get("failures", [])),
        check("config_export", config_export, export_data.get("exported"), export_data.get("failures", [])),
        readiness_check("deployment_readiness", deployment_readiness, deployment_data),
        readiness_check("sim_readiness", sim_readiness, sim_data) if sim_readiness else missing_check("sim_readiness"),
        room_check(room_report, read_json(room_report)) if room_report else missing_check("room_map"),
        live_safety_check(live_safety, read_json(live_safety)) if live_safety else missing_check("live_hardware_safety"),
    ]
    if hardware_blockers:
        checks.append(hardware_blockers_check(hardware_blockers, read_json(hardware_blockers)))
    failures = unique_failures(failure for item in checks for failure in item["failures"])
    return {
        "transfer_approved": not failures,
        "checks": checks,
        "summary": {"passed": sum(1 for item in checks if item["passed"]), "total": len(checks), "failures": failures},
        "safety": "Transfer gate is evidence only; live hardware still requires manual preflight and supervision.",
    }


def check(name: str, path: Path, passed: object, failures: object) -> dict[str, Any]:
    valid_failures = validated_failures(failures, name)
    valid_pass = passed is True and not valid_failures
    if not valid_pass and not valid_failures:
        valid_failures = [f"{name}_failed"]
    return {"name": name, "path": str(path), "passed": valid_pass, "failures": valid_failures}


def validated_failures(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        return [f"{name}_invalid_failures"]
    return list(value)


def missing_check(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "path": None,
        "passed": False,
        "failures": [f"{name}_missing"],
    }


def unique_failures(failures) -> list[str]:
    unique = []
    seen = set()
    for failure in failures:
        if failure not in seen:
            unique.append(failure)
            seen.add(failure)
    return unique


def readiness_check(name: str, path: Path, report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    records = report.get("records")
    record_failures, derived_ready, derived_blocked = validate_readiness_records(records)
    total = exact_nonnegative_int(summary.get("total"))
    ready = exact_nonnegative_int(summary.get("ready"))
    blocked = exact_nonnegative_int(summary.get("blocked"))
    summary_valid = (
        total is not None
        and ready is not None
        and blocked is not None
        and total == len(records or [])
        and ready == derived_ready
        and blocked == derived_blocked
        and ready + blocked == total
    )
    if name == "deployment_readiness":
        authority_failures = deployment_contract_failures(report)
    elif name == "sim_readiness" and (
        report.get("evidence_scope") != DESKTOP_DEVELOPMENT_SCOPE
        or report.get("deployment_authority") is not False
    ):
        authority_failures = ["desktop_scope_invalid"]
    else:
        authority_failures = []
    declared_failures = validated_failures(summary.get("failures", []), name)
    consistency_failures = record_failures + declared_failures + ([] if summary_valid else ["invalid_readiness_summary"])
    blocked_failures = (
        readiness_failures(report) if summary_valid and blocked else []
    )
    if authority_failures or consistency_failures:
        return {
            "name": name,
            "path": str(path),
            "passed": False,
            "failures": unique_failures(
                [*authority_failures, *consistency_failures, *blocked_failures]
            ),
            "ready": ready or 0,
            "total": total or 0,
            "blocked": blocked or 0,
            "evidence_scope": report.get("evidence_scope"),
            "deployment_authority": False,
        }
    assert total is not None and ready is not None and blocked is not None
    passed = total > 0 and blocked == 0 and ready == total
    failures = [] if passed else readiness_failures(report)
    return {
        "name": name,
        "path": str(path),
        "passed": passed,
        "failures": failures,
        "ready": ready,
        "total": total,
        "blocked": blocked,
        "deployment_authority": False,
    }


def validate_readiness_records(records: object) -> tuple[list[str], int, int]:
    if not isinstance(records, list):
        return ["invalid_readiness_records"], 0, 0
    failures: list[str] = []
    ready = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict) or type(record.get("ready")) is not bool:
            failures.append(f"record_{index}:invalid_ready")
            continue
        if not isinstance(record.get("task"), str) or not record["task"]:
            failures.append(f"record_{index}:invalid_task")
        raw_failures = record.get("failures")
        if not isinstance(raw_failures, list) or not all(
            isinstance(item, str) and item for item in raw_failures
        ):
            failures.append(f"record_{index}:invalid_failures")
            continue
        if record["ready"] is True:
            ready += 1
            if raw_failures:
                failures.append(f"record_{index}:ready_with_failures")
    return failures, ready, len(records) - ready


def readiness_failures(report: dict[str, Any]) -> list[str]:
    failures = []
    for record in report.get("records", []):
        if record.get("ready") is not True:
            task = record.get("task", "unknown")
            record_failures = record.get("failures", []) or ["not_ready"]
            failures.extend(f"{task}:{failure}" for failure in record_failures)
    return failures or ["readiness_blocked"]


def room_check(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    failures = validated_failures(summary.get("failures", []), "room_map")
    estimate = report.get("room_estimate", {})
    point_count = exact_nonnegative_int(summary.get("point_count"))
    dimensions = [finite_positive(estimate.get(key)) for key in ("width_m", "depth_m", "height_m")] if isinstance(estimate, dict) else []
    if point_count is None or point_count == 0 or len(dimensions) != 3 or any(value is None for value in dimensions):
        failures.append("room_map_invalid_metadata")
    ready = summary.get("mapping_ready") is True and not failures
    failures = [] if ready else failures or ["room_map_not_ready"]
    return {
        "name": "room_map",
        "path": str(path),
        "passed": ready,
        "failures": failures,
        "point_count": point_count,
        "width_m": dimensions[0] if len(dimensions) == 3 else None,
        "depth_m": dimensions[1] if len(dimensions) == 3 else None,
        "height_m": dimensions[2] if len(dimensions) == 3 else None,
    }


def live_safety_check(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    failures = validated_failures(summary.get("failures", []), "live_hardware_safety")
    checked = exact_nonnegative_int(summary.get("checked"))
    hardware_scripts = exact_nonnegative_int(summary.get("hardware_scripts"))
    learned_scripts = exact_nonnegative_int(summary.get("learned_checkpoint_hardware_scripts"))
    if (
        checked is None
        or checked == 0
        or hardware_scripts is None
        or learned_scripts is None
        or learned_scripts > hardware_scripts
        or hardware_scripts > checked
    ):
        failures.append("live_hardware_safety_invalid_summary")
    passed = summary.get("passed") is True and not failures
    failures = [] if passed else failures or ["live_hardware_safety_failed"]
    return {
        "name": "live_hardware_safety",
        "path": str(path),
        "passed": passed,
        "failures": failures,
        "hardware_scripts": hardware_scripts,
        "learned_checkpoint_hardware_scripts": learned_scripts,
    }


def hardware_blockers_check(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    blockers = failure_strings(report.get("blockers"))
    if blockers is None:
        blockers = ["hardware_blockers_invalid"]
    return {
        "name": "hardware_blockers",
        "path": str(path),
        "passed": not blockers,
        "failures": blockers,
        "blockers": blockers,
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
    data = json.loads(path.read_text()) if path else {}
    if not isinstance(data, dict):
        raise ValueError(f"evidence report must be a JSON object: {path}")
    return data


def report_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary")
    return summary if isinstance(summary, dict) else {}


def finite_positive(value: object) -> float | None:
    parsed = finite_number(value)
    return parsed if parsed is not None and parsed > 0.0 else None
