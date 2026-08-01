from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pytest

from flightrl.puffer4_edge_collection_evidence import (
    canonical_edge_environment_config,
    edge_environment_config_sha256,
)
from flightrl.puffer4_edge_replay import write_edge_passive_replay
from flightrl.puffer4_door_runner import native_build_marker_path
from flightrl.puffer4_edge_sequence import (
    load_edge_sequence_dataset,
    write_edge_sequence_dataset,
)
from flightrl.puffer4_edge_shadow_bundle import build_edge_shadow_bundle
from puffer4_edge_shadow_support import shadow_artifacts


@pytest.fixture(autouse=True)
def _stable_puffer_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flightrl.puffer4_door_runner.require_clean_puffer_revision",
        lambda _root: {"git_commit": "a" * 40},
    )


@pytest.mark.parametrize(
    ("field", "seed"),
    (("base_seed", 31_001), ("appearance_seed", 61_001)),
)
def test_shadow_replay_seeds_must_be_disjoint_from_evaluation_profiles(
    tmp_path: Path,
    field: str,
    seed: int,
) -> None:
    artifacts = shadow_artifacts(tmp_path)
    final = load_edge_sequence_dataset(artifacts.final)
    final.metadata[field] = seed
    config = canonical_edge_environment_config(
        environment=final.metadata["environment"],
        agents=final.metadata["agents"],
        base_seed=final.metadata["base_seed"],
        appearance_seed=final.metadata["appearance_seed"],
        collection_profile=final.metadata["collection_profile"],
    )
    final.metadata["environment_config"] = config
    final.metadata["environment_config_sha256"] = edge_environment_config_sha256(
        config
    )
    dataset = write_edge_sequence_dataset(tmp_path / f"overlap-{field}.npz", final)
    replay = tmp_path / f"overlap-{field}.jsonl"
    write_edge_passive_replay(
        checkpoint=artifacts.checkpoint,
        dataset=dataset,
        output=replay,
    )

    with pytest.raises(ValueError, match="overlap closed-loop evaluation"):
        build_edge_shadow_bundle(
            checkpoint=artifacts.checkpoint,
            evaluation_report=artifacts.evaluation,
            replay=replay,
        )


def test_shadow_evaluation_rejects_rebound_noncanonical_full_config(
    tmp_path: Path,
) -> None:
    artifacts = shadow_artifacts(tmp_path)
    assert artifacts.evaluation is not None
    report = json.loads(artifacts.evaluation.read_text())
    config = Path(report["environment_config_identity"]["path"])
    config.write_text(config.read_text() + "\n[forged]\nvalue = 1\n")
    from flightrl.evidence_scope import file_identity

    report["environment_config_identity"] = file_identity(config)
    artifacts.evaluation.write_text(json.dumps(report) + "\n")

    with pytest.raises(ValueError, match="not canonical"):
        build_edge_shadow_bundle(
            checkpoint=artifacts.checkpoint,
            evaluation_report=artifacts.evaluation,
            replay=artifacts.replay,
        )


def test_shadow_evaluation_rejects_profile_inconsistent_native_groups(
    tmp_path: Path,
) -> None:
    artifacts = shadow_artifacts(tmp_path)
    assert artifacts.evaluation is not None
    report = json.loads(artifacts.evaluation.read_text())
    metrics = report["profiles"]["clean"]["metrics"]
    metrics["low_light_episode_fraction"] = 0.25
    metrics["low_light_success_fraction"] = 60.0 / metrics["episodes"]
    artifacts.evaluation.write_text(json.dumps(report) + "\n")

    with pytest.raises(ValueError, match="low-light profile"):
        build_edge_shadow_bundle(
            checkpoint=artifacts.checkpoint,
            evaluation_report=artifacts.evaluation,
            replay=artifacts.replay,
        )


def test_shadow_evaluation_requires_full_checkpoint_native_identity(
    tmp_path: Path,
) -> None:
    artifacts = shadow_artifacts(tmp_path)
    assert artifacts.evaluation is not None and artifacts.native_root is not None
    report = json.loads(artifacts.evaluation.read_text())
    extension = Path(report["native_build_fingerprint"]["extension"]["path"])
    extension.write_bytes(b"different but internally fingerprinted extension")
    extension_sha256 = hashlib.sha256(extension.read_bytes()).hexdigest()
    report["native_build_fingerprint"]["extension"]["sha256"] = extension_sha256
    marker = native_build_marker_path(artifacts.native_root)
    marker.write_text(json.dumps(report["native_build_fingerprint"]) + "\n")
    artifacts.evaluation.write_text(json.dumps(report) + "\n")

    with pytest.raises(ValueError, match="do not match"):
        build_edge_shadow_bundle(
            checkpoint=artifacts.checkpoint,
            evaluation_report=artifacts.evaluation,
            replay=artifacts.replay,
        )
