from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    transfer_approved = bool(gate.get("transfer_approved", False))
    sim_by_task = {record["task"]: record for record in sim.get("records", [])}
    deploy_by_task = {record["task"]: record for record in deploy.get("records", [])}
    records = [
        checkpoint_record(task, sim_record, deploy_by_task.get(task, {}), transfer_approved)
        for task, sim_record in sorted(sim_by_task.items())
    ]
    hardware_approved = [record for record in records if record["hardware_approved"]]
    return {
        "transfer_gate": str(transfer_gate),
        "sim_readiness": str(sim_readiness),
        "deployment_readiness": str(deployment_readiness),
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
            "hardware_approved": len(hardware_approved),
        },
        "safety": "Only hardware_approved checkpoints may be considered for supervised live tests.",
    }


def checkpoint_record(task: str, sim: dict[str, Any], deploy: dict[str, Any], transfer_approved: bool) -> dict[str, Any]:
    checkpoint = sim.get("checkpoint") or deploy.get("checkpoint")
    deployment_ready = bool(deploy.get("ready", False))
    hardware_approved = transfer_approved and deployment_ready
    return {
        "task": task,
        "label": sim.get("label") or deploy.get("label"),
        "checkpoint": checkpoint,
        "tasks": sim.get("tasks") or deploy.get("tasks") or [task],
        "sim_ready": bool(sim.get("ready", False)),
        "sim_failures": sim.get("failures", []),
        "deployment_ready": deployment_ready,
        "deployment_failures": deploy.get("failures", ["missing_deployment_record"] if not deploy else []),
        "hardware_approved": hardware_approved,
        "status": status(bool(sim.get("ready", False)), deployment_ready, hardware_approved),
    }


def status(sim_ready: bool, deployment_ready: bool, hardware_approved: bool) -> str:
    if hardware_approved:
        return "hardware_approved"
    if deployment_ready:
        return "deployment_ready_but_transfer_blocked"
    if sim_ready:
        return "sim_only"
    return "blocked"


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
