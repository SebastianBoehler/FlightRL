from __future__ import annotations

import json
import subprocess
import sys

import pytest

from flightrl.sim2real.hardware_approval import HardwareApprovalError, hardware_approval_status, require_hardware_approved
from flightrl.sim2real.provenance import path_provenance


def test_require_hardware_approved_accepts_manifest_record(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_text("placeholder")
    manifest = write_manifest(tmp_path, checkpoint, hardware_approved=True)

    record = require_hardware_approved(checkpoint, manifest)

    assert record["checkpoint"] == str(checkpoint)


def test_require_hardware_approved_rejects_sim_only_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_text("placeholder")
    manifest = write_manifest(tmp_path, checkpoint, hardware_approved=False, transfer_approved=True, summary_hardware_approved=1)

    with pytest.raises(HardwareApprovalError, match="not hardware-approved"):
        require_hardware_approved(checkpoint, manifest)


def test_require_hardware_approved_rejects_manifest_when_transfer_gate_blocked(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_text("placeholder")
    manifest = write_manifest(tmp_path, checkpoint, hardware_approved=True, transfer_approved=False)

    with pytest.raises(HardwareApprovalError, match="transfer gate is not approved"):
        require_hardware_approved(checkpoint, manifest)


def test_require_hardware_approved_rejects_inconsistent_record(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_text("placeholder")
    manifest = write_manifest(tmp_path, checkpoint, hardware_approved=True, deployment_ready=False)

    with pytest.raises(HardwareApprovalError, match="approval is inconsistent"):
        require_hardware_approved(checkpoint, manifest)


def test_require_hardware_approved_rejects_manifest_without_provenance(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_text("placeholder")
    manifest = write_manifest(tmp_path, checkpoint, hardware_approved=True, include_provenance=False)

    with pytest.raises(HardwareApprovalError, match="missing input provenance"):
        require_hardware_approved(checkpoint, manifest)


def test_require_hardware_approved_rejects_stale_manifest_input(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_text("placeholder")
    manifest = write_manifest(tmp_path, checkpoint, hardware_approved=True)
    (tmp_path / "transfer_gate.json").write_text('{"transfer_approved": false}')

    with pytest.raises(HardwareApprovalError, match="input provenance is stale"):
        require_hardware_approved(checkpoint, manifest)


def test_hardware_approval_status_marks_monitor_only_blocked_without_raising(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_text("placeholder")
    manifest = write_manifest(tmp_path, checkpoint, hardware_approved=False, transfer_approved=True, summary_hardware_approved=1)

    status = hardware_approval_status(checkpoint, manifest)

    assert status["hardware_approved"] is False
    assert status["approval_status"] == "blocked"
    assert "not hardware-approved" in status["approval_error"]


def test_hardware_approval_status_includes_approved_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_text("placeholder")
    manifest = write_manifest(tmp_path, checkpoint, hardware_approved=True)

    status = hardware_approval_status(checkpoint, manifest)

    assert status["hardware_approved"] is True
    assert status["approval_status"] == "approved"
    assert status["approval_task"] == "obstacle_avoidance"
    assert status["approval_label"] == "candidate"


def test_avoidance_live_policy_exits_before_cflib_when_checkpoint_unapproved(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    manifest = write_manifest(tmp_path, checkpoint, hardware_approved=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/crazyflie_avoidance_policy.py",
            "--controller",
            "policy",
            "--checkpoint",
            str(checkpoint),
            "--approval-manifest",
            str(manifest),
            "--confirm-flight",
            "--duration-s",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "hardware approval blocked" in result.stderr
    assert "cflib" not in result.stderr.lower()


def test_hold_live_policy_exits_before_cflib_when_checkpoint_unapproved(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    manifest = write_manifest(tmp_path, checkpoint, hardware_approved=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/crazyflie_hold_policy.py",
            "--checkpoint",
            str(checkpoint),
            "--approval-manifest",
            str(manifest),
            "--confirm-flight",
            "--duration-s",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "hardware approval blocked" in result.stderr
    assert "cflib" not in result.stderr.lower()


def write_manifest(
    tmp_path,
    checkpoint,
    *,
    hardware_approved: bool,
    transfer_approved: bool | None = None,
    deployment_ready: bool | None = None,
    summary_hardware_approved: int | None = None,
    include_provenance: bool = True,
):
    manifest = tmp_path / "manifest.json"
    transfer_approved = hardware_approved if transfer_approved is None else transfer_approved
    deployment_ready = hardware_approved if deployment_ready is None else deployment_ready
    summary_hardware_approved = (1 if hardware_approved else 0) if summary_hardware_approved is None else summary_hardware_approved
    transfer = tmp_path / "transfer_gate.json"
    sim = tmp_path / "sim_readiness.json"
    deploy = tmp_path / "deployment_readiness.json"
    transfer.write_text(json.dumps({"transfer_approved": transfer_approved}))
    sim.write_text(json.dumps({"summary": {"ready": 1}, "records": []}))
    deploy.write_text(json.dumps({"summary": {"ready": 1}, "records": []}))
    data = {
        "transfer_approved": transfer_approved,
        "summary": {"hardware_approved": summary_hardware_approved},
        "records": [
            {
                "task": "obstacle_avoidance",
                "label": "candidate",
                "checkpoint": str(checkpoint),
                "status": "hardware_approved" if hardware_approved else "sim_only",
                "deployment_failures": [] if hardware_approved else ["replay_comparison"],
                "deployment_ready": deployment_ready,
                "hardware_approved": hardware_approved,
            }
        ],
    }
    if include_provenance:
        data["inputs"] = {
            "transfer_gate": path_provenance(transfer),
            "sim_readiness": path_provenance(sim),
            "deployment_readiness": path_provenance(deploy),
        }
    manifest.write_text(
        json.dumps(data)
    )
    return manifest
