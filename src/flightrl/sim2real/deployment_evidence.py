from __future__ import annotations

from flightrl.evidence_scope import (
    EDGE_DEPLOYMENT_VERIFIER_MISSING,
    deployment_claim_failures,
    require_file_identity,
)
from flightrl.sixdof.controller import CONTROLLERS
from flightrl.sixdof.tasks import MULTITASK, TASKS


SCHEMA = "flightrl.edge_v3.deployment_readiness.v1"
TARGET = "ai_deck_gap8"


def deployment_contract_failures(report: object) -> list[str]:
    if not isinstance(report, dict):
        return ["deployment_report_invalid"]
    failures = deployment_claim_failures(report)
    if failures:
        return failures
    if report.get("schema") != SCHEMA:
        failures.append("deployment_schema_invalid")
    if report.get("target") != TARGET:
        failures.append("deployment_target_invalid")
    records = report.get("records")
    if not isinstance(records, list):
        return [*failures, "deployment_records_invalid"]
    seen_tasks = set()
    for index, record in enumerate(records):
        failures.extend(record_contract_failures(record, index, seen_tasks))
    failures.append(EDGE_DEPLOYMENT_VERIFIER_MISSING)
    return failures


def record_contract_failures(
    record: object,
    index: int,
    seen_tasks: set[str],
) -> list[str]:
    prefix = f"record_{index}"
    if not isinstance(record, dict):
        return [f"{prefix}:invalid"]
    failures = []
    task = record.get("task")
    tasks = record.get("tasks")
    if (
        task not in (*TASKS, MULTITASK)
        or task in seen_tasks
        or not isinstance(tasks, list)
        or not tasks
        or not all(isinstance(item, str) for item in tasks)
        or len(tasks) != len(set(tasks))
        or any(item not in TASKS for item in tasks)
        or (task != MULTITASK and task not in tasks)
    ):
        failures.append(f"{prefix}:task_contract_invalid")
    elif isinstance(task, str):
        seen_tasks.add(task)
    if record.get("controller") not in CONTROLLERS:
        failures.append(f"{prefix}:controller_invalid")
    failures.extend(identity_failures(record, prefix, "checkpoint"))
    failures.extend(identity_failures(record, prefix, "bundle"))
    return failures


def identity_failures(record: dict, prefix: str, field: str) -> list[str]:
    path = record.get(field)
    if not isinstance(path, str) or not path:
        return [f"{prefix}:{field}_missing"]
    try:
        require_file_identity(
            record.get(f"{field}_identity"),
            path,
            label=f"deployment {field}",
        )
    except (OSError, ValueError):
        return [f"{prefix}:{field}_identity_invalid"]
    return []
