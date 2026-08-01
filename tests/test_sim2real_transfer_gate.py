from __future__ import annotations

import json
from pathlib import Path

from flightrl.evidence_scope import EDGE_DEPLOYMENT_VERIFIER_MISSING
from flightrl.sim2real.transfer_gate import build_transfer_gate
from sim2real_transfer_gate_test_support import valid_inputs, write_json


def test_transfer_gate_blocks_typed_metadata_without_edge_verifier(tmp_path: Path) -> None:
    report = build_transfer_gate(**valid_inputs(tmp_path))

    assert report["transfer_approved"] is False
    assert report["summary"] == {
        "passed": 6,
        "total": 7,
        "failures": [EDGE_DEPLOYMENT_VERIFIER_MISSING],
    }


def test_transfer_gate_blocks_missing_sim_room_and_live_safety(tmp_path: Path) -> None:
    inputs = valid_inputs(tmp_path)
    for name in ("sim_readiness", "room_report", "live_safety"):
        del inputs[name]

    report = build_transfer_gate(**inputs)

    assert report["transfer_approved"] is False
    assert {
        "sim_readiness_missing",
        "room_map_missing",
        "live_hardware_safety_missing",
    }.issubset(report["summary"]["failures"])


def test_transfer_gate_blocks_failed_evidence(tmp_path: Path) -> None:
    inputs = valid_inputs(tmp_path)
    inputs["audit"] = write_json(
        tmp_path / "audit.json",
        {"transfer_ready": False, "blocking_items": ["motor_bench_failed"]},
    )
    readiness = json.loads(inputs["deployment_readiness"].read_text())
    readiness["summary"] = {"total": 1, "ready": 0, "blocked": 1}
    readiness["records"][0]["ready"] = False
    readiness["records"][0]["failures"] = ["replay"]
    inputs["deployment_readiness"] = write_json(
        tmp_path / "readiness.json", readiness
    )

    report = build_transfer_gate(**inputs)

    assert report["transfer_approved"] is False
    assert "motor_bench_failed" in report["summary"]["failures"]
    assert "obstacle_avoidance:replay" in report["summary"]["failures"]


def test_transfer_gate_rejects_truthy_strings_and_pass_with_failures(
    tmp_path: Path,
) -> None:
    inputs = valid_inputs(tmp_path)
    inputs["audit"] = write_json(
        tmp_path / "audit.json",
        {"transfer_ready": "true", "blocking_items": []},
    )
    inputs["profile"] = write_json(
        tmp_path / "profile.json",
        {"summary": {"profile_ready": "true", "failures": []}},
    )
    inputs["config_export"] = write_json(
        tmp_path / "export.json",
        {"exported": True, "failures": ["stale_export"]},
    )

    report = build_transfer_gate(**inputs)

    assert report["transfer_approved"] is False
    assert [item["passed"] for item in report["checks"][:3]] == [False] * 3
    assert "stale_export" in report["summary"]["failures"]


def test_transfer_gate_rejects_inconsistent_readiness_summary(
    tmp_path: Path,
) -> None:
    inputs = valid_inputs(tmp_path)
    readiness = json.loads(inputs["deployment_readiness"].read_text())
    readiness["records"][0]["ready"] = False
    readiness["records"][0]["failures"] = ["failed"]
    inputs["deployment_readiness"] = write_json(
        tmp_path / "readiness.json", readiness
    )

    report = build_transfer_gate(**inputs)

    assert report["transfer_approved"] is False
    assert "invalid_readiness_summary" in report["summary"]["failures"]
