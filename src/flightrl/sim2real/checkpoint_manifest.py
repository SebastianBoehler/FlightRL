from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightrl.evidence_scope import EDGE_DEPLOYMENT_SCOPE, deployment_authority_failures
from flightrl.sim2real.provenance import path_provenance


def build_checkpoint_manifest(
    *,
    transfer_gate: Path,
    sim_readiness: Path,
    deployment_readiness: Path,
) -> dict[str, Any]:
    gate = read_json(transfer_gate)
    sim = read_json(sim_readiness)
    deploy = read_json(deployment_readiness)
    authority_failures = deployment_authority_failures(deploy)
    transfer_approved = gate.get("transfer_approved") is True and not authority_failures
    sim_by_task = {record["task"]: record for record in sim.get("records", [])}
    deploy_by_task = {record["task"]: record for record in deploy.get("records", [])}
    records = [
        checkpoint_record(task, sim_record, deploy_by_task.get(task, {}), transfer_approved, authority_failures)
        for task, sim_record in sorted(sim_by_task.items())
    ]
    return {
        "transfer_gate": str(transfer_gate),
        "sim_readiness": str(sim_readiness),
        "deployment_readiness": str(deployment_readiness),
        "evidence_scope": EDGE_DEPLOYMENT_SCOPE if not authority_failures else deploy.get("evidence_scope"),
        "deployment_authority": not authority_failures,
        "deployment_authority_failures": authority_failures,
        "inputs": {
            "transfer_gate": path_provenance(transfer_gate),
            "sim_readiness": path_provenance(sim_readiness),
            "deployment_readiness": path_provenance(deployment_readiness),
        },
        "transfer_approved": transfer_approved,
        "records": records,
        "summary": {
            "total": len(records),
            "sim_ready": sum(1 for record in records if record["sim_ready"]),
            "deployment_ready": sum(1 for record in records if record["deployment_ready"]),
            "hardware_approved": 0,
        },
        "safety": "This generic manifest cannot authorize learned live control. A typed edge-v3 deployment bundle is required.",
    }


def checkpoint_record(
    task: str,
    sim: dict[str, Any],
    deploy: dict[str, Any],
    transfer_approved: bool,
    authority_failures: list[str] | None = None,
) -> dict[str, Any]:
    sim_failures = record_failures(sim)
    deploy_failures = record_failures(deploy)
    sim_ready = sim.get("ready") is True and not sim_failures
    checkpoint = sim.get("checkpoint") or deploy.get("checkpoint")
    authority_failures = authority_failures or []
    checkpoint_provenance = path_provenance(Path(checkpoint)) if checkpoint else {}
    checkpoint_failures = [] if checkpoint_provenance.get("sha256") else ["checkpoint_missing"]
    if not same_checkpoint(sim.get("checkpoint"), deploy.get("checkpoint")):
        checkpoint_failures.append("checkpoint_identity_mismatch")
    deployment_failures = list(authority_failures)
    deployment_failures.extend(
        failure
        for failure in (deploy_failures if deploy else ["missing_deployment_record"])
        if failure not in deployment_failures
    )
    deployment_failures.extend(
        failure for failure in checkpoint_failures if failure not in deployment_failures
    )
    deployment_ready = not authority_failures and not checkpoint_failures and deploy.get("ready") is True and not deploy_failures
    return {
        "task": task,
        "label": sim.get("label") or deploy.get("label"),
        "checkpoint": checkpoint,
        "checkpoint_provenance": checkpoint_provenance,
        "tasks": sim.get("tasks") or deploy.get("tasks") or [task],
        "sim_ready": sim_ready,
        "sim_failures": sim_failures,
        "deployment_ready": deployment_ready,
        "deployment_failures": deployment_failures,
        "hardware_approved": False,
        "hardware_blocker": "edge_v3_deployment_bundle_required",
        "status": status(sim_ready, deployment_ready, transfer_approved),
    }


def status(sim_ready: bool, deployment_ready: bool, transfer_approved: bool) -> str:
    if deployment_ready:
        if not sim_ready:
            return "deployment_ready_but_sim_blocked"
        if not transfer_approved:
            return "deployment_ready_but_transfer_blocked"
        return "deployment_ready_but_edge_bundle_required"
    if sim_ready:
        return "sim_only"
    return "blocked"


def same_checkpoint(sim_checkpoint: object, deploy_checkpoint: object) -> bool:
    if not sim_checkpoint or not deploy_checkpoint:
        return False
    try:
        return Path(str(sim_checkpoint)).resolve() == Path(str(deploy_checkpoint)).resolve()
    except OSError:
        return str(sim_checkpoint) == str(deploy_checkpoint)


def record_failures(record: dict[str, Any]) -> list[str]:
    failures = record.get("failures", [])
    if not isinstance(failures, list) or not all(isinstance(item, str) for item in failures):
        return ["invalid_failures"]
    if record and record.get("ready") is not True and record.get("ready") is not False:
        return [*failures, "invalid_ready"]
    return failures


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sim-To-Real Checkpoint Manifest",
        "",
        f"- Transfer approved: `{report['transfer_approved']}`",
        f"- Hardware-approved checkpoints: `{report['summary']['hardware_approved']}`",
        "",
        "| task | label | status | sim ready | deployment ready | hardware approved | failures | checkpoint |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for record in report["records"]:
        failures = record["deployment_failures"] or record["sim_failures"]
        lines.append(
            f"| {record['task']} | {record['label']} | {record['status']} | {record['sim_ready']} | "
            f"{record['deployment_ready']} | {record['hardware_approved']} | {', '.join(failures) or 'none'} | `{record['checkpoint']}` |"
        )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
