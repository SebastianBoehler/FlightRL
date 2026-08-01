from __future__ import annotations

import json

import pytest

from flightrl.evidence_scope import file_identity
from flightrl.puffer4_edge_training import (
    EDGE_TRAINING_REPORT_SCHEMA,
    EdgeTrainingRejected,
)
from puffer4_edge_artifact_support import native_build_fingerprint, write_sequence
import scripts.train_puffer_edge_student as training_cli


def test_rejected_training_writes_bound_report_without_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    fingerprint = native_build_fingerprint(tmp_path / "native-build")
    train = write_sequence(tmp_path / "train.npz", "train", 11, fingerprint)
    selection = write_sequence(
        tmp_path / "selection.npz", "selection", 21, fingerprint
    )
    checkpoint = tmp_path / "student.pt"
    report_path = tmp_path / "training.json"
    rejection_report = {
        "schema": EDGE_TRAINING_REPORT_SCHEMA,
        "status": "rejected",
        "history": [{"epoch": 1}],
        "baselines": {"previous_action": {}, "constant_grounding": {}},
        "baseline_gate": {
            "passed": False,
            "checks": {"previous_action": False, "constant_grounding": True},
            "failed_checks": ["previous_action"],
        },
    }

    def reject(*_args, **_kwargs):
        raise EdgeTrainingRejected(rejection_report)

    def checkpoint_must_not_be_built(*_args, **_kwargs):
        raise AssertionError("rejected training must not build a checkpoint")

    monkeypatch.setattr(training_cli, "train_edge_student", reject)
    monkeypatch.setattr(
        training_cli,
        "build_edge_checkpoint_payload",
        checkpoint_must_not_be_built,
    )

    status = training_cli.main(
        [
            "--train-dataset",
            str(train),
            "--selection-dataset",
            str(selection),
            "--checkpoint",
            str(checkpoint),
            "--training-report",
            str(report_path),
            "--epochs",
            "1",
        ]
    )

    assert status != 0
    assert checkpoint.exists() is False
    report = json.loads(report_path.read_text())
    assert report["status"] == "rejected"
    assert report["datasets"] == {
        "train": file_identity(train),
        "selection": file_identity(selection),
    }
    assert report["native_build_fingerprint"] == fingerprint
    assert set(report["source_identity"]) == {
        "script",
        "artifact_paths",
        "trainer",
        "policy",
        "training_data",
        "training_report",
        "selection_gate",
        "state_digest",
        "native_identity",
        "source_identity",
    }


@pytest.mark.parametrize(
    ("mutation_stage", "report_written"),
    (("dataset_load", False), ("checkpoint_build", True)),
)
def test_training_source_mutation_blocks_checkpoint(
    tmp_path,
    monkeypatch,
    mutation_stage: str,
    report_written: bool,
) -> None:
    fingerprint = native_build_fingerprint(tmp_path / "native-build")
    train = write_sequence(tmp_path / "train.npz", "train", 11, fingerprint)
    selection = write_sequence(
        tmp_path / "selection.npz", "selection", 21, fingerprint
    )
    checkpoint = tmp_path / "student.pt"
    report_path = tmp_path / "training.json"
    original_identity = training_cli.edge_training_source_identity()
    changed_identity = {
        **original_identity,
        "script": {**original_identity["script"], "sha256": "0" * 64},
    }
    mutated = False
    original_loader = training_cli.load_edge_sequence_dataset

    def source_identity():
        return changed_identity if mutated else original_identity

    def load_dataset(*args, **kwargs):
        nonlocal mutated
        dataset = original_loader(*args, **kwargs)
        if mutation_stage == "dataset_load":
            mutated = True
        return dataset

    def build_checkpoint(*_args, **_kwargs):
        nonlocal mutated
        if mutation_stage == "checkpoint_build":
            mutated = True
        return {}

    def save_must_not_run(*_args, **_kwargs):
        raise AssertionError("mutated training sources must block checkpoint save")

    monkeypatch.setattr(training_cli, "edge_training_source_identity", source_identity)
    monkeypatch.setattr(training_cli, "load_edge_sequence_dataset", load_dataset)
    monkeypatch.setattr(
        training_cli,
        "train_edge_student",
        lambda *_args, **_kwargs: (object(), {}),
    )
    monkeypatch.setattr(
        training_cli,
        "build_edge_checkpoint_payload",
        build_checkpoint,
    )
    monkeypatch.setattr(training_cli, "save_edge_checkpoint", save_must_not_run)

    with pytest.raises(RuntimeError, match="training sources changed"):
        training_cli.main(
            [
                "--train-dataset",
                str(train),
                "--selection-dataset",
                str(selection),
                "--checkpoint",
                str(checkpoint),
                "--training-report",
                str(report_path),
                "--epochs",
                "1",
            ]
        )

    assert checkpoint.exists() is False
    assert report_path.exists() is report_written
