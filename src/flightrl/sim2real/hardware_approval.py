from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightrl.sim2real.provenance import verify_path_provenance


class HardwareApprovalError(RuntimeError):
    pass


def require_hardware_approved(checkpoint: str | Path, manifest: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest)
    if not manifest_path.exists():
        raise HardwareApprovalError(f"hardware approval manifest not found: {manifest_path}")
    report = json.loads(manifest_path.read_text())
    validate_manifest_provenance(report)
    validate_manifest_approved(report)
    record = find_checkpoint_record(report, checkpoint)
    if record is None:
        raise HardwareApprovalError(f"checkpoint is not listed in hardware approval manifest: {checkpoint}")
    validate_record_approved(record)
    return record


def validate_manifest_provenance(report: dict[str, Any]) -> None:
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        raise HardwareApprovalError("hardware approval manifest is missing input provenance")
    for name, expected in inputs.items():
        if not isinstance(expected, dict) or "path" not in expected:
            raise HardwareApprovalError(f"hardware approval manifest has invalid input provenance: {name}")
        result = verify_path_provenance(expected)
        if not result["passed"]:
            raise HardwareApprovalError(
                f"hardware approval manifest input provenance is stale: {name} failure={result['failure']}"
            )


def validate_manifest_approved(report: dict[str, Any]) -> None:
    if not report.get("transfer_approved", False):
        raise HardwareApprovalError("manifest transfer gate is not approved")
    if int(report.get("summary", {}).get("hardware_approved", 0) or 0) < 1:
        raise HardwareApprovalError("manifest contains no hardware-approved checkpoints")


def validate_record_approved(record: dict[str, Any]) -> None:
    if not record.get("hardware_approved", False):
        failures = record.get("deployment_failures") or record.get("sim_failures") or ["not_hardware_approved"]
        raise HardwareApprovalError(
            f"checkpoint is not hardware-approved: task={record.get('task')} status={record.get('status')} failures={','.join(failures)}"
        )
    if not record.get("deployment_ready", False):
        raise HardwareApprovalError(f"checkpoint approval is inconsistent: task={record.get('task')} deployment_ready=False")
    if record.get("status") != "hardware_approved":
        raise HardwareApprovalError(f"checkpoint approval is inconsistent: task={record.get('task')} status={record.get('status')}")


def hardware_approval_status(checkpoint: str | Path, manifest: str | Path) -> dict[str, Any]:
    try:
        record = require_hardware_approved(checkpoint, manifest)
    except HardwareApprovalError as exc:
        return {
            "hardware_approved": False,
            "approval_status": "blocked",
            "approval_error": str(exc),
            "approval_manifest": str(manifest),
        }
    return {
        "hardware_approved": True,
        "approval_status": "approved",
        "approval_error": "",
        "approval_manifest": str(manifest),
        "approval_task": record.get("task"),
        "approval_label": record.get("label"),
    }


def find_checkpoint_record(report: dict[str, Any], checkpoint: str | Path) -> dict[str, Any] | None:
    target = normalize_path(checkpoint)
    for record in report.get("records", []):
        if normalize_path(record.get("checkpoint")) == target:
            return record
    return None


def normalize_path(path: str | Path | None) -> str:
    if path is None:
        return ""
    value = Path(path)
    try:
        return str(value.resolve())
    except OSError:
        return str(value)
