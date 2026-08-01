from __future__ import annotations

import json

import pytest

from flightrl.puffer4_door_readiness import build_fixed_door_yaw_readiness
from flightrl.semantic.readiness import (
    build_bounded_forward_readiness,
    build_yaw_only_readiness,
    load_yaw_only_readiness,
    write_readiness,
)
from test_puffer4_door_readiness import _evidence as fixed_door_evidence
from test_semantic_bounded_authority import _training_report


NO_AXES = {
    "vx_body_m_s": False,
    "vy_body_m_s": False,
    "vz_m_s": False,
    "yawrate_deg_s": False,
}
YAW_ONLY_AXES = NO_AXES | {"yawrate_deg_s": True}
BOUNDED_FORWARD_AXES = YAW_ONLY_AXES | {"vx_body_m_s": True}


def test_failed_fixed_door_gate_serializes_candidate_but_grants_nothing(
    tmp_path,
) -> None:
    checkpoint, simulation, shadow, shadow_csv = fixed_door_evidence(tmp_path)
    summary = json.loads(shadow.read_text())
    summary["checkpoint"] = str((tmp_path / "other.bin").resolve())
    shadow.write_text(json.dumps(summary))

    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )

    assert report["schema_version"] == 2
    assert report.get("candidate_authority") == "yaw_only"
    assert report.get("candidate_axis_authority") == YAW_ONLY_AXES
    assert report["approved_authority"] == "none"
    assert report["axis_authority"] == NO_AXES
    assert report["next_live_gate_passed"] is False


def test_passing_fixed_door_gate_grants_only_candidate_axes(tmp_path) -> None:
    checkpoint, simulation, shadow, shadow_csv = fixed_door_evidence(tmp_path)

    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )

    assert report["schema_version"] == 2
    assert report.get("candidate_authority") == "yaw_only"
    assert report.get("candidate_axis_authority") == YAW_ONLY_AXES
    assert report["approved_authority"] == "yaw_only"
    assert report["axis_authority"] == YAW_ONLY_AXES
    assert report["next_live_gate_passed"] is True


def test_failed_generic_yaw_gate_serializes_no_effective_authority(
    tmp_path,
) -> None:
    checkpoint, simulation, replay = _yaw_evidence(tmp_path, passed=False)

    report = build_yaw_only_readiness(checkpoint, simulation, replay)

    assert report["schema_version"] == 2
    assert report.get("candidate_authority") == "yaw_only"
    assert report.get("candidate_axis_authority") == YAW_ONLY_AXES
    assert report["approved_authority"] == "none"
    assert report["axis_authority"] == NO_AXES
    assert report["next_live_gate_passed"] is False
    assert "remains disabled" in report["reason"]


def test_failed_bounded_gate_serializes_no_effective_authority(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    training = tmp_path / "training.json"
    replay = tmp_path / "replay.json"
    training.write_text(json.dumps(_training_report(checkpoint, gate=False)))
    replay.write_text(json.dumps({"translation_shadow_gate_passed": True}))

    report = build_bounded_forward_readiness(checkpoint, training, replay)

    assert report["schema_version"] == 2
    assert report.get("candidate_authority") == "bounded_forward_yaw"
    assert report.get("candidate_axis_authority") == BOUNDED_FORWARD_AXES
    assert report["approved_authority"] == "none"
    assert report["axis_authority"] == NO_AXES
    assert report["next_live_gate_passed"] is False


def test_loader_rejects_forged_v2_effective_authority_after_failed_gate(
    tmp_path,
) -> None:
    checkpoint, simulation, replay = _yaw_evidence(tmp_path, passed=False)
    report = build_yaw_only_readiness(checkpoint, simulation, replay)
    report["schema_version"] = 2
    report["candidate_authority"] = "yaw_only"
    report["candidate_axis_authority"] = YAW_ONLY_AXES
    report["approved_authority"] = "yaw_only"
    report["axis_authority"] = YAW_ONLY_AXES
    report_path = write_readiness(tmp_path / "readiness.json", report)

    with pytest.raises(ValueError, match="effective authority contradicts gate"):
        load_yaw_only_readiness(report_path, checkpoint)


def test_loader_preserves_passing_schema_v1_compatibility(tmp_path) -> None:
    checkpoint, simulation, replay = _yaw_evidence(tmp_path, passed=True)
    report = build_yaw_only_readiness(checkpoint, simulation, replay)
    report["schema_version"] = 1
    report.pop("candidate_authority", None)
    report.pop("candidate_axis_authority", None)
    report_path = write_readiness(tmp_path / "legacy-readiness.json", report)

    loaded = load_yaw_only_readiness(report_path, checkpoint)

    assert loaded["approved_authority"] == "yaw_only"
    assert loaded["axis_authority"] == YAW_ONLY_AXES


def _yaw_evidence(tmp_path, *, passed: bool):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    simulation = tmp_path / "simulation.json"
    simulation.write_text(
        json.dumps(
            {
                "collision_rate": 0.0 if passed else 0.01,
                "preacquisition_horizontal_p95_m_s": 0.02,
                "visible_yaw_sign_accuracy": 0.95,
                "visible_abs_yawrate_p95_deg_s": 8.0,
                "max_abs_yawrate_deg_s": 15.0,
                "success_rate": 0.8,
            }
        )
    )
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"next_live_shadow_gate_passed": True}))
    return checkpoint, simulation, replay
