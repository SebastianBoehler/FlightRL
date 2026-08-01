from __future__ import annotations

import json
import pytest

from flightrl.sim2real.hardware_approval import (
    EDGE_BUNDLE_REQUIRED,
    HardwareApprovalError,
    hardware_approval_status,
    require_hardware_approved,
)


def test_generic_manifest_cannot_authorize_learned_live_control(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_text("checkpoint")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(forged_positive_manifest(checkpoint)))

    with pytest.raises(HardwareApprovalError, match="typed edge-v3 deployment bundle"):
        require_hardware_approved(checkpoint, manifest)


def test_hardware_approval_status_is_not_evaluated_without_manifest(tmp_path) -> None:
    status = hardware_approval_status(tmp_path / "policy.pt", None)

    assert status["hardware_approved"] is False
    assert status["approval_status"] == "not_evaluated"


def test_hardware_approval_status_blocks_even_forged_positive_manifest(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    status = hardware_approval_status(tmp_path / "policy.pt", manifest)

    assert status["hardware_approved"] is False
    assert status["approval_status"] == "blocked"
    assert status["approval_error"] == EDGE_BUNDLE_REQUIRED

def forged_positive_manifest(checkpoint) -> dict:
    return {
        "evidence_scope": "edge_deployment",
        "deployment_authority": True,
        "transfer_approved": True,
        "summary": {"hardware_approved": 1},
        "records": [
            {
                "checkpoint": str(checkpoint),
                "sim_ready": True,
                "deployment_ready": True,
                "hardware_approved": True,
                "status": "hardware_approved",
            }
        ],
    }
