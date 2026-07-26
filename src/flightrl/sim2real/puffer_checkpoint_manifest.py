from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightrl.sim2real.checkpoint_manifest import status
from flightrl.sim2real.provenance import path_provenance


def build_puffer_checkpoint_manifest(*, transfer_gate: Path, bundle_report: Path) -> dict[str, Any]:
    gate = read_json(transfer_gate)
    bundle_data = read_json(bundle_report)
    bundle = bundle_data.get("bundle", {})
    transfer_approved = bool(gate.get("transfer_approved", False))
    records = puffer_records(bundle, transfer_approved)
    return {
        "transfer_gate": str(transfer_gate),
        "bundle_report": str(bundle_report),
        "inputs": {
            "transfer_gate": path_provenance(transfer_gate),
            "bundle_report": path_provenance(bundle_report),
        },
        "transfer_approved": transfer_approved,
        "transfer_failures": transfer_failures(gate),
        "records": records,
        "summary": {
            "total": len(records),
            "sim_ready": sum(1 for record in records if record["sim_ready"]),
            "deployment_ready": sum(1 for record in records if record["deployment_ready"]),
            "hardware_approved": sum(1 for record in records if record["hardware_approved"]),
        },
        "safety": "Only hardware_approved Puffer checkpoints may be considered for supervised live tests.",
    }


def puffer_records(bundle: dict[str, Any], transfer_approved: bool) -> list[dict[str, Any]]:
    label = str(bundle.get("label", "puffer_bundle"))
    velocity = bundle.get("velocity", {})
    obstacle_checkpoint = bundle.get("obstacle_checkpoint")
    velocity_checkpoint = bundle.get("velocity_checkpoint")
    velocity_ready = (
        bool(velocity_checkpoint)
        and bool(velocity)
        and all(item.get("gate", {}).get("passed", False) for item in velocity.values())
    )
    obstacle_ready = bool(obstacle_checkpoint) and bool(bundle.get("obstacle", {}).get("passed", False))
    return [
        puffer_record(
            task="obstacle_avoidance",
            label=f"{label}:obstacle",
            checkpoint=obstacle_checkpoint,
            ready=obstacle_ready,
            failures=component_failures(obstacle_ready, obstacle_checkpoint, "puffer_obstacle_transfer_failed"),
            transfer_approved=transfer_approved,
        ),
        puffer_record(
            task="velocity_target",
            label=f"{label}:velocity",
            checkpoint=velocity_checkpoint,
            ready=velocity_ready,
            failures=component_failures(velocity_ready, velocity_checkpoint, "puffer_velocity_transfer_failed"),
            transfer_approved=transfer_approved,
        ),
    ]


def component_failures(ready: bool, checkpoint: str | None, failure: str) -> list[str]:
    if ready:
        return []
    failures = []
    if not checkpoint:
        failures.append("missing_checkpoint")
    failures.append(failure)
    return failures


def puffer_record(
    *,
    task: str,
    label: str,
    checkpoint: str | None,
    ready: bool,
    failures: list[str],
    transfer_approved: bool,
) -> dict[str, Any]:
    hardware_approved = transfer_approved and ready
    return {
        "task": task,
        "label": label,
        "checkpoint": checkpoint,
        "tasks": [task],
        "sim_ready": ready,
        "sim_failures": failures,
        "deployment_ready": ready,
        "deployment_failures": failures,
        "hardware_approved": hardware_approved,
        "status": status(ready, ready, hardware_approved),
    }


def render_markdown(report: dict[str, Any]) -> str:
    transfer_failures = ", ".join(report.get("transfer_failures", [])) or "none"
    lines = [
        "# Puffer Checkpoint Manifest",
        "",
        f"- Transfer approved: `{report['transfer_approved']}`",
        f"- Transfer failures: `{transfer_failures}`",
        f"- Hardware-approved checkpoints: `{report['summary']['hardware_approved']}`",
        "",
        "| task | label | status | deployment ready | hardware approved | failures | checkpoint |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for record in report["records"]:
        failures = record["deployment_failures"] or record["sim_failures"]
        lines.append(
            f"| {record['task']} | {record['label']} | {record['status']} | {record['deployment_ready']} | "
            f"{record['hardware_approved']} | {', '.join(failures) or 'none'} | `{record['checkpoint']}` |"
        )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def transfer_failures(gate: dict[str, Any]) -> list[str]:
    summary = gate.get("summary", {})
    if isinstance(summary.get("failures"), list):
        return [str(item) for item in summary["failures"]]
    failures = []
    for check in gate.get("checks", []):
        failures.extend(str(item) for item in check.get("failures", []))
    return sorted(dict.fromkeys(failures))
