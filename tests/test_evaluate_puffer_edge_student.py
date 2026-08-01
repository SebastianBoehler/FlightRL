from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import flightrl.puffer4_edge_sequence as edge_sequence
from flightrl.evidence_scope import file_identity
from flightrl.puffer4_edge_training import EDGE_TRAINING_REPORT_SCHEMA
from puffer4_edge_artifact_support import checkpoint_artifacts
import scripts.evaluate_puffer_edge_student as evaluation_cli


@pytest.mark.parametrize(
    "output_target",
    ("checkpoint", "environment_config", "training", "train", "selection"),
)
def test_evaluator_rejects_output_aliases_before_evaluation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    output_target: str,
) -> None:
    checkpoint = tmp_path / "student.pt"
    training = tmp_path / "training.json"
    train = tmp_path / "train.npz"
    selection = tmp_path / "selection.npz"
    environment_config = tmp_path / "config" / "edge-env.ini"
    environment_config.parent.mkdir()
    files = {
        "checkpoint": checkpoint,
        "training": training,
        "train": train,
        "selection": selection,
        "environment_config": environment_config,
    }
    for name, path in files.items():
        path.write_bytes(name.encode())
    original = {path: path.read_bytes() for path in files.values()}
    metadata = SimpleNamespace(
        trained_target_ids=(0,),
        training_identity=file_identity(training),
        native_build_fingerprint={},
        hidden_size=48,
        policy_contract_sha256="a" * 64,
    )
    report = {
        "datasets": {
            "train": file_identity(train),
            "selection": file_identity(selection),
        }
    }
    monkeypatch.setattr(
        evaluation_cli,
        "load_edge_checkpoint",
        lambda _path: (object(), metadata),
    )
    monkeypatch.setattr(
        evaluation_cli,
        "_load_training_report",
        lambda _identity: report,
    )
    monkeypatch.setattr(
        evaluation_cli,
        "_require_final_seed_disjointness",
        lambda _report: (_ for _ in ()).throw(
            AssertionError("evaluation work must not start for aliased artifacts")
        ),
    )

    with pytest.raises(ValueError, match="artifact paths must be distinct"):
        evaluation_cli.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--puffer-root",
                str(tmp_path),
                "--env-name",
                "edge-env",
                "--output",
                str(files[output_target]),
            ]
        )

    assert {path: path.read_bytes() for path in files.values()} == original


def test_evaluator_only_reuses_the_checkpoint_bound_native_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"native": "fingerprint"}
    calls = []

    def verify(root, env_name, native_files):
        calls.append((root, env_name, native_files))
        return expected

    monkeypatch.setattr(evaluation_cli, "verify_native_build", verify)

    assert evaluation_cli._verify_build(Path("/puffer"), "edge-env") is expected
    assert calls == [
        (Path("/puffer"), "edge-env", evaluation_cli.EDGE_STUDENT_NATIVE_FILES)
    ]
    assert "build_environment" not in vars(evaluation_cli)
    assert "export_edge_student_assets" not in vars(evaluation_cli)


def test_evaluator_loads_only_the_current_training_report_schema(tmp_path) -> None:
    report = tmp_path / "training.json"
    report.write_text(json.dumps({"schema": EDGE_TRAINING_REPORT_SCHEMA}))

    assert evaluation_cli._load_training_report(file_identity(report)) == {
        "schema": EDGE_TRAINING_REPORT_SCHEMA
    }

    report.write_text(json.dumps({"schema": "flightrl.edge_v3.training_report.v2"}))
    with pytest.raises(ValueError, match="schema is incompatible"):
        evaluation_cli._load_training_report(file_identity(report))


def test_evaluator_seed_check_does_not_repeat_execution_trace(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = checkpoint_artifacts(tmp_path)
    training = json.loads(artifacts.training.read_text())
    calls = []
    original = edge_sequence.require_edge_execution_trace

    def counted(dataset) -> None:
        calls.append(dataset.metadata["split"])
        original(dataset)

    monkeypatch.setattr(edge_sequence, "require_edge_execution_trace", counted)

    evaluation_cli._require_final_seed_disjointness(training)

    assert calls == []


@pytest.mark.parametrize(
    ("mutated", "match"),
    (
        ("checkpoint", "checkpoint changed during evaluation"),
        ("config", "environment config changed during evaluation"),
    ),
)
def test_evaluator_rejects_bound_file_mutation_before_report(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    mutated: str,
    match: str,
) -> None:
    checkpoint_path = tmp_path / "student.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    config_path = (
        tmp_path / "config" / "flightrl_edge_v3_door_student.ini"
    )
    config_path.parent.mkdir()
    config_path.write_text("config\n")
    output = tmp_path / "evaluation.json"
    calls = 0

    def evaluate(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == len(evaluation_cli.EDGE_EVALUATION_PROFILES):
            target = checkpoint_path if mutated == "checkpoint" else config_path
            target.write_bytes(b"changed")
        return {"metrics": {}, "gate": {"passed": True}}

    _mock_evaluator_runtime(
        monkeypatch,
        checkpoint_path,
        config_path,
        evaluate,
    )
    monkeypatch.setattr(
        evaluation_cli,
        "require_evaluation_metric_consistency",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match=match):
        evaluation_cli.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--puffer-root",
                str(tmp_path),
                "--agents",
                "1",
                "--steps",
                "1",
                "--output",
                str(output),
            ]
        )

    assert calls == len(evaluation_cli.EDGE_EVALUATION_PROFILES)
    assert output.exists() is False


def test_evaluator_rejects_inconsistent_profile_before_report(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "student.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    config_path = (
        tmp_path / "config" / "flightrl_edge_v3_door_student.ini"
    )
    config_path.parent.mkdir()
    config_path.write_text("config\n")
    output = tmp_path / "evaluation.json"
    _mock_evaluator_runtime(
        monkeypatch,
        checkpoint_path,
        config_path,
        lambda *_args, **_kwargs: {
            "metrics": {},
            "gate": {"passed": True},
        },
    )

    with pytest.raises(ValueError, match="episode count is inconsistent"):
        evaluation_cli.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--puffer-root",
                str(tmp_path),
                "--agents",
                "1",
                "--steps",
                "1",
                "--output",
                str(output),
            ]
        )

    assert output.exists() is False


def _mock_evaluator_runtime(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
    config_path: Path,
    evaluate,
) -> None:
    fingerprint = {"native": "fingerprint"}
    training_report = checkpoint_path.with_name("training.json")
    training_report.write_text("{}\n")
    metadata = SimpleNamespace(
        trained_target_ids=(0,),
        training_identity=file_identity(training_report),
        native_build_fingerprint=fingerprint,
        hidden_size=48,
        policy_contract_sha256="a" * 64,
    )
    monkeypatch.setattr(
        evaluation_cli,
        "load_edge_checkpoint",
        lambda _path: (object(), metadata),
    )
    monkeypatch.setattr(evaluation_cli, "_load_training_report", lambda _identity: {})
    monkeypatch.setattr(evaluation_cli, "_training_dataset_paths", lambda _report: {})
    monkeypatch.setattr(evaluation_cli, "_require_final_seed_disjointness", lambda _: None)
    monkeypatch.setattr(evaluation_cli, "_verify_build", lambda *_args: fingerprint)
    monkeypatch.setattr(
        evaluation_cli,
        "require_matching_edge_native_build_fingerprints",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        evaluation_cli,
        "require_current_edge_native_build_fingerprint",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluation_cli,
        "write_edge_student_config",
        lambda *_args: config_path,
    )
    monkeypatch.setattr(
        evaluation_cli,
        "load_puffer",
        lambda *_args: ({"env": {}, "vec": {}}, object()),
    )
    monkeypatch.setattr(evaluation_cli, "evaluate_edge_student", evaluate)
