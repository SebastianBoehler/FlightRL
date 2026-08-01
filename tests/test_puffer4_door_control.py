from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from fixed_door_promotion_fixture import write_test_promotion
from flightrl import puffer4_door_control as control
from flightrl.puffer4_door_control import (
    DoorPufferControlAdapter,
    load_readiness_bound_control_adapter,
)
from flightrl.puffer4_door_contract import CORRECTED_DOOR_ACTION_CONTRACT
from flightrl.puffer4_door_policy_contract import door_policy_contract_report
from flightrl.puffer4_door_runtime import DoorPufferRuntime
from flightrl.semantic.readiness import file_sha256


def _telemetry() -> dict[str, float]:
    return {
        "stateEstimate.x": 0.0,
        "stateEstimate.y": 0.0,
        "stateEstimate.z": 0.8,
        "stateEstimate.vx": 0.0,
        "stateEstimate.vy": 0.0,
        "stateEstimate.vz": 0.0,
        "stateEstimate.roll": 0.0,
        "stateEstimate.pitch": 0.0,
        "stateEstimate.yaw": 0.0,
        "gyro.x": 0.0,
        "gyro.y": 0.0,
        "gyro.z": 0.0,
    }


def test_door_control_adapter_verifies_report_and_scales_output(tmp_path) -> None:
    checkpoint = tmp_path / "door.bin"
    torch.save(DoorPufferRuntime(hidden_size=32).state_dict(), checkpoint)
    report = tmp_path / "door.json"
    report.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "action_contract": CORRECTED_DOOR_ACTION_CONTRACT.to_report(),
                "policy_contract": door_policy_contract_report(
                    hidden_size=32,
                    num_layers=1,
                ),
            }
        )
    )
    adapter = DoorPufferControlAdapter.from_evaluation_report(checkpoint, report)

    output = adapter.step(
        frame=np.full((48, 64), 51, dtype=np.uint8),
        telemetry=_telemetry(),
        prompt="interior door",
        detection=None,
    )

    assert output["vx_body_m_s"] == pytest.approx(
        output["action_forward"] * 0.55
    )
    assert output["yawrate_deg_s"] == pytest.approx(
        output["action_yaw"] * 70.0
    )


def test_door_control_adapter_records_bounded_executed_action(tmp_path) -> None:
    checkpoint = tmp_path / "door.bin"
    torch.save(DoorPufferRuntime(hidden_size=32).state_dict(), checkpoint)
    adapter = DoorPufferControlAdapter(
        checkpoint,
        action_contract=CORRECTED_DOOR_ACTION_CONTRACT,
    )

    adapter.record_executed_action(
        vx_body_m_s=0.0,
        yawrate_deg_s=7.0,
    )
    adapter.step(
        frame=np.zeros((48, 64), dtype=np.uint8),
        telemetry=_telemetry(),
        prompt="door",
        detection=None,
    )

    assert adapter.runtime.previous_action == pytest.approx((0.0, 0.1))


def test_door_control_adapter_rejects_other_semantic_targets(tmp_path) -> None:
    checkpoint = tmp_path / "door.bin"
    torch.save(DoorPufferRuntime(hidden_size=32).state_dict(), checkpoint)
    adapter = DoorPufferControlAdapter(
        checkpoint,
        action_contract=CORRECTED_DOOR_ACTION_CONTRACT,
    )

    with pytest.raises(ValueError, match="door target"):
        adapter.step(
            frame=np.zeros((48, 64), dtype=np.uint8),
            telemetry=_telemetry(),
            prompt="monitor",
            detection=None,
        )


def test_door_control_adapter_requires_action_contract_in_report(tmp_path) -> None:
    checkpoint = tmp_path / "door.bin"
    torch.save(DoorPufferRuntime(hidden_size=32).state_dict(), checkpoint)
    report = tmp_path / "door.json"
    report.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            }
        )
    )

    with pytest.raises(ValueError, match="action contract"):
        DoorPufferControlAdapter.from_evaluation_report(checkpoint, report)


def test_door_control_adapter_requires_matching_policy_contract(tmp_path) -> None:
    checkpoint = tmp_path / "door.bin"
    torch.save(DoorPufferRuntime(hidden_size=32).state_dict(), checkpoint)
    report = tmp_path / "door.json"
    report.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "action_contract": CORRECTED_DOOR_ACTION_CONTRACT.to_report(),
            }
        )
    )

    with pytest.raises(ValueError, match="policy contract"):
        DoorPufferControlAdapter.from_evaluation_report(checkpoint, report)


def test_door_control_adapter_rejects_checkpoint_architecture_mismatch(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "door.bin"
    torch.save(DoorPufferRuntime(hidden_size=32).state_dict(), checkpoint)
    report = tmp_path / "door.json"
    report.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "action_contract": CORRECTED_DOOR_ACTION_CONTRACT.to_report(),
                "policy_contract": door_policy_contract_report(
                    hidden_size=64,
                    num_layers=1,
                ),
            }
        )
    )

    with pytest.raises(ValueError, match="architecture"):
        DoorPufferControlAdapter.from_evaluation_report(checkpoint, report)


def test_live_control_rejects_evidence_changed_after_readiness_load(
    tmp_path,
) -> None:
    checkpoint, report = write_test_promotion(tmp_path)
    adapter = DoorPufferControlAdapter.from_evaluation_report(
        checkpoint,
        report,
    )
    bundle = adapter.bundle
    assert bundle is not None
    readiness = {
        "checkpoint": str(bundle.checkpoint_path),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "evaluation_report": str(bundle.report_path),
        "evaluation_report_sha256": bundle.report_sha256,
        "lineage_report": str(bundle.lineage_report_path),
        "lineage_report_sha256": bundle.lineage_report_sha256,
    }
    report.write_text(report.read_text() + "\n")
    assert file_sha256(report) != readiness["evaluation_report_sha256"]

    with pytest.raises(ValueError, match="evaluation report SHA-256"):
        load_readiness_bound_control_adapter(
            checkpoint,
            report,
            readiness,
        )


def test_live_control_revalidates_evidence_after_policy_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint, report = write_test_promotion(tmp_path)
    bundle = DoorPufferControlAdapter.from_evaluation_report(
        checkpoint,
        report,
    ).bundle
    assert bundle is not None
    readiness = {
        "checkpoint": str(bundle.checkpoint_path),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "evaluation_report": str(bundle.report_path),
        "evaluation_report_sha256": bundle.report_sha256,
        "lineage_report": str(bundle.lineage_report_path),
        "lineage_report_sha256": bundle.lineage_report_sha256,
    }
    original = control.load_fixed_door_checkpoint_snapshot

    def mutate_after_snapshot(path, expected_sha256):
        snapshot = original(path, expected_sha256)
        checkpoint.write_bytes(b"changed after snapshot")
        return snapshot

    monkeypatch.setattr(
        control,
        "load_fixed_door_checkpoint_snapshot",
        mutate_after_snapshot,
    )

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        load_readiness_bound_control_adapter(
            checkpoint,
            report,
            readiness,
        )
