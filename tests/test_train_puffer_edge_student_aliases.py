from __future__ import annotations

import pytest

import scripts.train_puffer_edge_student as training_cli


@pytest.mark.parametrize(
    ("checkpoint_name", "report_name"),
    (
        ("train.npz", "training.json"),
        ("selection.npz", "training.json"),
        ("student.pt", "train.npz"),
        ("student.pt", "selection.npz"),
        ("student.pt", "student.pt"),
    ),
)
def test_training_rejects_output_aliases_before_loading_inputs(
    tmp_path,
    monkeypatch,
    checkpoint_name: str,
    report_name: str,
) -> None:
    train = tmp_path / "train.npz"
    selection = tmp_path / "selection.npz"
    train.write_bytes(b"train-dataset")
    selection.write_bytes(b"selection-dataset")
    original = {train: train.read_bytes(), selection: selection.read_bytes()}

    def work_must_not_start():
        raise AssertionError("training work must not start for aliased artifacts")

    monkeypatch.setattr(
        training_cli,
        "edge_training_source_identity",
        work_must_not_start,
    )

    with pytest.raises(ValueError, match="artifact paths must be distinct"):
        training_cli.main(
            [
                "--train-dataset",
                str(train),
                "--selection-dataset",
                str(selection),
                "--checkpoint",
                str(tmp_path / checkpoint_name),
                "--training-report",
                str(tmp_path / report_name),
            ]
        )

    assert {path: path.read_bytes() for path in original} == original


def test_training_rejects_existing_hardlink_output_alias(tmp_path, monkeypatch) -> None:
    train = tmp_path / "train.npz"
    selection = tmp_path / "selection.npz"
    checkpoint = tmp_path / "student.pt"
    train.write_bytes(b"train-dataset")
    selection.write_bytes(b"selection-dataset")
    checkpoint.hardlink_to(train)
    original = train.read_bytes()
    monkeypatch.setattr(
        training_cli,
        "edge_training_source_identity",
        lambda: (_ for _ in ()).throw(
            AssertionError("training work must not start for aliased artifacts")
        ),
    )

    with pytest.raises(ValueError, match="artifact paths must be distinct"):
        training_cli.main(
            [
                "--train-dataset",
                str(train),
                "--selection-dataset",
                str(selection),
                "--checkpoint",
                str(checkpoint),
                "--training-report",
                str(tmp_path / "training.json"),
            ]
        )

    assert train.read_bytes() == checkpoint.read_bytes() == original
