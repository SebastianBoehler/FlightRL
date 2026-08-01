from __future__ import annotations

from pathlib import Path

import pytest

import flightrl.puffer4_edge_sequence as edge_sequence
from flightrl.puffer4_edge_shadow_bundle import build_edge_shadow_bundle
from puffer4_edge_shadow_support import shadow_artifacts


@pytest.fixture(autouse=True)
def _stable_puffer_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flightrl.puffer4_door_runner.require_clean_puffer_revision",
        lambda _root: {"git_commit": "a" * 40},
    )


def test_bundle_validation_replays_each_dataset_trace_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = shadow_artifacts(tmp_path)
    assert artifacts.evaluation is not None
    assert artifacts.replay is not None
    calls = []
    original = edge_sequence.require_edge_execution_trace

    def counted(dataset) -> None:
        calls.append(dataset.metadata["split"])
        original(dataset)

    monkeypatch.setattr(edge_sequence, "require_edge_execution_trace", counted)

    build_edge_shadow_bundle(
        checkpoint=artifacts.checkpoint,
        evaluation_report=artifacts.evaluation,
        replay=artifacts.replay,
    )

    assert calls == ["train", "selection", "final"]
