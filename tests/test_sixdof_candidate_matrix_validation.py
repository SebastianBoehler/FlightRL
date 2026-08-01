from __future__ import annotations

import json
from pathlib import Path

import pytest

from sixdof_candidate_matrix_test_support import (
    MATRIX,
    identity,
    suite_record,
    write_checkpoint,
)


def test_candidate_matrix_rejects_desktop_evidence_for_other_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    other = tmp_path / "other.pt"
    write_checkpoint(checkpoint)
    write_checkpoint(other)
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"records": [suite_record("candidate", checkpoint)]}))
    parity = {
        "schema": "flightrl.sixdof.desktop_export.v1",
        "evidence_scope": "desktop_cpu_only",
        "deployment_authority": False,
        "checkpoint": identity(other),
        "model": identity(other),
        "parity": {"max_abs_error": 0.0},
    }

    with pytest.raises(ValueError, match="desktop parity checkpoint"):
        MATRIX.checkpoint_records(
            suite,
            {"candidate": parity},
            {},
            1.0e-5,
        )


@pytest.mark.parametrize("kind", ("parity", "latency"))
def test_candidate_matrix_rejects_supplied_empty_desktop_report(
    tmp_path: Path,
    kind: str,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    write_checkpoint(checkpoint)
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"records": [suite_record("candidate", checkpoint)]}))
    parity = {"candidate": {}} if kind == "parity" else {}
    latency = {"candidate": {}} if kind == "latency" else {}

    with pytest.raises(ValueError, match=f"desktop {kind}"):
        MATRIX.checkpoint_records(suite, parity, latency, 1.0e-5)


def test_candidate_matrix_rejects_coerced_gate_and_nonfinite_metrics(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    write_checkpoint(checkpoint)
    suite = tmp_path / "suite.json"
    invalid_gate = suite_record("candidate", checkpoint)
    invalid_gate["gate"]["passed"] = "true"
    suite.write_text(json.dumps({"records": [invalid_gate]}))
    with pytest.raises(ValueError, match="gate"):
        MATRIX.checkpoint_records(suite, {}, {}, 1.0e-5)

    invalid_metric = suite_record("candidate", checkpoint)
    invalid_metric["metrics"]["mean_completed_fraction"] = float("nan")
    suite.write_text(json.dumps({"records": [invalid_metric]}))
    with pytest.raises(ValueError, match="metrics"):
        MATRIX.checkpoint_records(suite, {}, {}, 1.0e-5)


def test_candidate_matrix_rejects_aggregate_pass_with_failed_task_gate(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    write_checkpoint(checkpoint)
    suite = tmp_path / "suite.json"
    contradictory = suite_record("candidate", checkpoint)
    contradictory["per_task_gate"]["position_yaw"] = {
        "passed": False,
        "failures": ["completion"],
    }
    suite.write_text(json.dumps({"records": [contradictory]}))

    with pytest.raises(ValueError, match="contradicts per-task"):
        MATRIX.checkpoint_records(suite, {}, {}, 1.0e-5)


def test_candidate_matrix_rejects_retired_controller(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    write_checkpoint(checkpoint)
    suite = tmp_path / "suite.json"
    legacy = suite_record("candidate", checkpoint)
    legacy["controller"] = "legacy_raw_action"
    suite.write_text(json.dumps({"records": [legacy]}))

    with pytest.raises(ValueError, match="controller"):
        MATRIX.checkpoint_records(suite, {}, {}, 1.0e-5)


@pytest.mark.parametrize("error", (float("nan"), float("inf"), -1.0))
def test_candidate_matrix_rejects_invalid_parity_metrics(error: float) -> None:
    with pytest.raises(ValueError, match="max_abs_error"):
        MATRIX.compact_parity({"parity": {"max_abs_error": error}}, 1.0e-5)


def test_candidate_matrix_does_not_fall_back_from_invalid_torchscript_latency() -> None:
    report = {
        "torchscript_result": {},
        "eager": {"per_sample_us": 3.0, "samples_per_second": 333333.0},
    }

    with pytest.raises(ValueError, match="latency metrics"):
        MATRIX.compact_latency(report)


def test_candidate_matrix_rejects_legacy_edge_evidence(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    write_checkpoint(checkpoint)
    suite = tmp_path / "suite.json"
    legacy = suite_record("candidate", checkpoint)
    legacy["edge_latency"] = {"present": True}
    suite.write_text(json.dumps({"records": [legacy]}))

    with pytest.raises(ValueError, match="legacy edge_latency"):
        MATRIX.checkpoint_records(suite, {}, {}, 1.0e-5)
