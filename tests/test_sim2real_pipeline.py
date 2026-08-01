from __future__ import annotations

from pathlib import Path

from flightrl.evidence_scope import EDGE_DEPLOYMENT_VERIFIER_MISSING
from flightrl.sim2real.pipeline import build_pipeline, output_paths, pipeline_summary
from sim2real_pipeline_test_support import write_ready_inputs


def test_pipeline_builds_artifact_chain_but_cannot_approve_edge_transfer(tmp_path: Path) -> None:
    paths = write_ready_inputs(tmp_path)
    outputs = output_paths(tmp_path / "out", "ready")

    report = build_pipeline(outputs=outputs, **paths)

    assert report["transfer_approved"] is False
    assert "deployment_readiness_blocked" in report["blocking_items"]
    assert report["hardware_approved_checkpoints"] == 0
    assert report["inputs"]["hardware_config"]["exists"] is True
    assert len(report["inputs"]["hardware_config"]["sha256"]) == 64
    assert report["inputs"]["hardware_config"]["size_bytes"] > 0
    assert report["inputs"]["live_scripts"][0]["exists"] is True
    assert len(report["inputs"]["live_scripts"][0]["sha256"]) == 64
    assert outputs["pipeline"].exists()


def test_pipeline_summary_rejects_coerced_readiness_values(tmp_path: Path) -> None:
    outputs = output_paths(tmp_path, "invalid")
    report = pipeline_summary(
        outputs,
        {"transfer_ready": "false", "blocking_items": []},
        {"summary": {"profile_ready": 1}},
        {"exported": "yes"},
        {"transfer_approved": 1, "summary": {"failures": []}},
        {"summary": {"hardware_approved": "1"}},
        {},
    )

    assert report["transfer_ready"] is False
    assert report["profile_ready"] is False
    assert report["config_exported"] is False
    assert report["transfer_approved"] is False
    assert report["hardware_approved_checkpoints"] == 0
    assert "evidence_gap" in report["artifacts"]


def test_pipeline_summary_rejects_pass_flags_with_declared_failures(tmp_path: Path) -> None:
    report = pipeline_summary(
        output_paths(tmp_path, "contradictory"),
        {"transfer_ready": True, "blocking_items": ["stale_audit"]},
        {"summary": {"profile_ready": True, "failures": ["stale_profile"]}},
        {"exported": True, "failures": ["stale_export"]},
        {"transfer_approved": True, "summary": {"failures": ["stale_gate"]}},
        {"summary": {"hardware_approved": 0}},
        {},
    )

    assert report["transfer_ready"] is False
    assert report["profile_ready"] is False
    assert report["config_exported"] is False
    assert report["transfer_approved"] is False
    assert report["gate_failures"] == [
        "stale_gate",
        EDGE_DEPLOYMENT_VERIFIER_MISSING,
    ]
