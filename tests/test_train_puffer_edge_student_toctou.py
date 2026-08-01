from __future__ import annotations

import json
from pathlib import Path

import pytest

from flightrl.evidence_scope import file_identity
from puffer4_edge_artifact_support import native_build_fingerprint, write_sequence
import scripts.train_puffer_edge_student as training_cli


@pytest.mark.parametrize(
    ("mutation_stage", "report_written"),
    (("dataset_load", False), ("checkpoint_build", True)),
)
def test_training_dataset_mutation_blocks_checkpoint(
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
    captured = {
        "train": file_identity(train),
        "selection": file_identity(selection),
    }
    original_loader = training_cli.load_edge_sequence_dataset
    mutated = False

    def mutate_train_dataset() -> None:
        nonlocal mutated
        if not mutated:
            with train.open("ab") as handle:
                handle.write(b"dataset-mutated-during-training")
            mutated = True

    def load_dataset(*args, **kwargs):
        dataset = original_loader(*args, **kwargs)
        if mutation_stage == "dataset_load":
            mutate_train_dataset()
        return dataset

    def build_checkpoint(*_args, **kwargs):
        assert Path(kwargs["dataset"]) == Path(captured["selection"]["path"])
        if mutation_stage == "checkpoint_build":
            mutate_train_dataset()
        return {}

    def save_must_not_run(*_args, **_kwargs):
        raise AssertionError("mutated training datasets must block checkpoint save")

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

    with pytest.raises(RuntimeError, match="training dataset changed"):
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
    if report_written:
        assert json.loads(report_path.read_text())["datasets"] == captured
