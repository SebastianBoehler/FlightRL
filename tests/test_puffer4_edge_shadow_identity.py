from __future__ import annotations

from pathlib import Path

import pytest

import flightrl.puffer4_edge_shadow_bundle as edge_shadow
from flightrl.puffer4_edge_shadow_bundle import (
    build_edge_shadow_bundle,
    require_edge_shadow_bundle,
    write_edge_shadow_bundle,
)
from puffer4_edge_shadow_support import shadow_artifacts
import scripts.build_edge_v3_shadow_bundle as shadow_cli


@pytest.fixture(autouse=True)
def _stable_puffer_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flightrl.puffer4_door_runner.require_clean_puffer_revision",
        lambda _root: {"git_commit": "a" * 40},
    )


@pytest.mark.parametrize("validation", ("producer", "consumer"))
@pytest.mark.parametrize("artifact", ("checkpoint", "evaluation", "replay"))
def test_shadow_bundle_rejects_input_mutation_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation: str,
    artifact: str,
) -> None:
    artifacts = shadow_artifacts(tmp_path)
    assert artifacts.evaluation is not None
    assert artifacts.replay is not None
    paths = {
        "checkpoint": artifacts.checkpoint,
        "evaluation": artifacts.evaluation,
        "replay": artifacts.replay,
    }
    bundle = None
    if validation == "consumer":
        bundle = build_edge_shadow_bundle(
            checkpoint=paths["checkpoint"],
            evaluation_report=paths["evaluation"],
            replay=paths["replay"],
        )
    original = edge_shadow._require_replay

    def mutate_after_replay(*args, **kwargs):
        result = original(*args, **kwargs)
        paths[artifact].write_bytes(b"mutated-during-shadow-validation")
        return result

    monkeypatch.setattr(edge_shadow, "_require_replay", mutate_after_replay)

    with pytest.raises(RuntimeError, match=f"{artifact} changed"):
        if validation == "producer":
            build_edge_shadow_bundle(
                checkpoint=paths["checkpoint"],
                evaluation_report=paths["evaluation"],
                replay=paths["replay"],
            )
        else:
            require_edge_shadow_bundle(bundle)


def test_shadow_bundle_writer_rejects_bound_input_as_output(tmp_path: Path) -> None:
    artifacts = shadow_artifacts(tmp_path)
    assert artifacts.evaluation is not None
    assert artifacts.replay is not None
    bundle = build_edge_shadow_bundle(
        checkpoint=artifacts.checkpoint,
        evaluation_report=artifacts.evaluation,
        replay=artifacts.replay,
    )
    checkpoint_bytes = artifacts.checkpoint.read_bytes()

    with pytest.raises(ValueError, match="artifact paths must be distinct"):
        write_edge_shadow_bundle(bundle, artifacts.checkpoint)

    assert artifacts.checkpoint.read_bytes() == checkpoint_bytes


def test_shadow_bundle_cli_rejects_output_alias_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    evaluation = tmp_path / "evaluation.json"
    replay = tmp_path / "replay.jsonl"
    for path in (checkpoint, evaluation, replay):
        path.write_bytes(b"input")

    def build_must_not_run(**_kwargs):
        raise AssertionError("bundle build must not start with aliased output")

    monkeypatch.setattr(shadow_cli, "build_edge_shadow_bundle", build_must_not_run)

    with pytest.raises(ValueError, match="artifact paths must be distinct"):
        shadow_cli.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--evaluation-report",
                str(evaluation),
                "--replay",
                str(replay),
                "--output",
                str(checkpoint),
            ]
        )
