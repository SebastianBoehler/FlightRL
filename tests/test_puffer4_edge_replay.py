from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from flightrl.evidence_scope import file_identity
import flightrl.puffer4_edge_replay as edge_replay
from flightrl.puffer4_edge_replay import (
    require_edge_passive_replay,
    write_edge_passive_replay,
)
from flightrl.puffer4_edge_sequence import (
    load_edge_sequence_dataset,
    write_edge_sequence_dataset,
)
from puffer4_edge_artifact_support import checkpoint_artifacts


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    artifacts = checkpoint_artifacts(tmp_path)
    return artifacts.checkpoint, artifacts.final


def _load_must_not_run(*_args, **_kwargs):
    raise AssertionError("replay alias guard must run before artifact loading")


def test_offline_passive_replay_is_complete_deterministic_and_non_actuating(
    tmp_path: Path,
) -> None:
    checkpoint, dataset = _artifacts(tmp_path)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    write_edge_passive_replay(checkpoint=checkpoint, dataset=dataset, output=first)
    write_edge_passive_replay(checkpoint=checkpoint, dataset=dataset, output=second)
    validated = require_edge_passive_replay(first)

    assert first.read_bytes() == second.read_bytes()
    assert validated["summary"]["complete"] is True
    assert validated["summary"]["controls_drone"] is False
    assert validated["header"]["trained_target_ids"] == [0]
    assert validated["header"]["checkpoint_identity"] == file_identity(checkpoint)
    assert validated["header"]["dataset_identity"] == file_identity(dataset)
    assert validated["header"]["native_build_fingerprint"][
        "dependency_revision"
    ] == {"git_commit": "a" * 40}


def test_offline_passive_replay_rejects_authority_tampering(tmp_path: Path) -> None:
    checkpoint, dataset = _artifacts(tmp_path)
    output = tmp_path / "replay.jsonl"
    write_edge_passive_replay(checkpoint=checkpoint, dataset=dataset, output=output)
    lines = output.read_text().splitlines()
    summary = json.loads(lines[-1])
    summary["controls_drone"] = True
    lines[-1] = json.dumps(summary)
    output.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="control authority"):
        require_edge_passive_replay(output)


def test_offline_passive_replay_rejects_prior_schema(tmp_path: Path) -> None:
    checkpoint, dataset = _artifacts(tmp_path)
    output = tmp_path / "replay.jsonl"
    write_edge_passive_replay(checkpoint=checkpoint, dataset=dataset, output=output)
    lines = output.read_text().splitlines()
    header = json.loads(lines[0])
    header["schema"] = "flightrl.edge_v3.offline_passive_replay.v1"
    lines[0] = json.dumps(header)
    output.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="schema"):
        require_edge_passive_replay(output)


def test_offline_passive_replay_rejects_full_native_identity_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint, dataset_path = _artifacts(tmp_path)
    dataset = load_edge_sequence_dataset(dataset_path)
    dataset.metadata["native_build_fingerprint"]["extension"]["sha256"] = "d" * 64
    mismatched = write_edge_sequence_dataset(tmp_path / "mismatched.npz", dataset)

    with pytest.raises(ValueError, match="do not match"):
        write_edge_passive_replay(
            checkpoint=checkpoint,
            dataset=mismatched,
            output=tmp_path / "replay.jsonl",
        )


@pytest.mark.parametrize("artifact", ("checkpoint", "dataset"))
@pytest.mark.parametrize("mutation_stage", ("load", "before_replace"))
def test_offline_passive_replay_input_mutation_blocks_atomic_replace(
    tmp_path: Path,
    monkeypatch,
    artifact: str,
    mutation_stage: str,
) -> None:
    checkpoint, dataset = _artifacts(tmp_path)
    target = checkpoint if artifact == "checkpoint" else dataset
    output = tmp_path / "replay.jsonl"
    original_output = b"existing-replay-must-survive"
    output.write_bytes(original_output)

    def mutate_target() -> None:
        with target.open("ab") as handle:
            handle.write(b"mutated-during-passive-replay")

    if mutation_stage == "load":
        loader_name = (
            "load_edge_checkpoint"
            if artifact == "checkpoint"
            else "load_edge_sequence_dataset"
        )
        original_loader = getattr(edge_replay, loader_name)

        def mutating_loader(*args, **kwargs):
            loaded = original_loader(*args, **kwargs)
            mutate_target()
            return loaded

        monkeypatch.setattr(edge_replay, loader_name, mutating_loader)
    else:
        original_write = edge_replay._write_record

        def mutating_write(handle, value):
            original_write(handle, value)
            if value.get("record") == "summary":
                mutate_target()

        monkeypatch.setattr(edge_replay, "_write_record", mutating_write)

    with pytest.raises(RuntimeError, match=f"{artifact} changed"):
        write_edge_passive_replay(
            checkpoint=checkpoint,
            dataset=dataset,
            output=output,
        )

    assert output.read_bytes() == original_output


@pytest.mark.parametrize("artifact", ("checkpoint", "dataset"))
@pytest.mark.parametrize("alias_kind", ("direct", "resolved"))
def test_offline_passive_replay_rejects_lexical_or_resolved_output_alias(
    tmp_path: Path,
    monkeypatch,
    artifact: str,
    alias_kind: str,
) -> None:
    checkpoint, dataset = _artifacts(tmp_path)
    source = checkpoint if artifact == "checkpoint" else dataset
    output = (
        source
        if alias_kind == "direct"
        else source.parent / "unused-directory" / ".." / source.name
    )
    before = source.read_bytes()
    monkeypatch.setattr(edge_replay, "load_edge_checkpoint", _load_must_not_run)
    monkeypatch.setattr(
        edge_replay,
        "load_edge_sequence_dataset",
        _load_must_not_run,
    )

    with pytest.raises(
        ValueError,
        match=f"{artifact} and output artifact paths must be distinct",
    ):
        write_edge_passive_replay(
            checkpoint=checkpoint,
            dataset=dataset,
            output=output,
        )

    assert source.read_bytes() == before


@pytest.mark.parametrize("artifact", ("checkpoint", "dataset"))
@pytest.mark.parametrize("link_kind", ("symlink", "hardlink"))
def test_offline_passive_replay_rejects_existing_file_alias(
    tmp_path: Path,
    monkeypatch,
    artifact: str,
    link_kind: str,
) -> None:
    checkpoint, dataset = _artifacts(tmp_path)
    source = checkpoint if artifact == "checkpoint" else dataset
    output = tmp_path / f"{artifact}-{link_kind}.jsonl"
    before = source.read_bytes()
    try:
        if link_kind == "symlink":
            output.symlink_to(source)
        else:
            os.link(source, output)
    except OSError as exc:
        pytest.skip(f"filesystem does not support {link_kind}: {exc}")
    monkeypatch.setattr(edge_replay, "load_edge_checkpoint", _load_must_not_run)
    monkeypatch.setattr(
        edge_replay,
        "load_edge_sequence_dataset",
        _load_must_not_run,
    )

    with pytest.raises(
        ValueError,
        match=f"{artifact} and output artifact paths must be distinct",
    ):
        write_edge_passive_replay(
            checkpoint=checkpoint,
            dataset=dataset,
            output=output,
        )

    assert source.read_bytes() == before


@pytest.mark.parametrize(
    ("artifact", "identity_field"),
    (
        ("selection", "dataset_identity"),
        ("training_report", "training_identity"),
    ),
)
def test_offline_passive_replay_rejects_bound_training_artifact_output_alias(
    tmp_path: Path,
    monkeypatch,
    artifact: str,
    identity_field: str,
) -> None:
    checkpoint, dataset = _artifacts(tmp_path)
    _actor, metadata = edge_replay.load_edge_checkpoint(checkpoint)
    source = Path(getattr(metadata, identity_field)["path"])
    before = source.read_bytes()
    monkeypatch.setattr(
        edge_replay,
        "load_edge_sequence_dataset",
        _load_must_not_run,
    )

    with pytest.raises(
        ValueError,
        match=f"{artifact} and output artifact paths must be distinct",
    ):
        write_edge_passive_replay(
            checkpoint=checkpoint,
            dataset=dataset,
            output=source,
        )

    assert source.read_bytes() == before
