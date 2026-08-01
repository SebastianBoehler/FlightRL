from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flightrl.puffer4_edge_collection_evidence import (
    canonical_edge_environment_config,
    edge_environment_config_sha256,
)
from flightrl.puffer4_edge_dagger import fixed_student_mask
from flightrl.puffer4_edge_sequence import (
    EdgeSequenceDataset,
    edge_dataset_metadata,
    load_edge_sequence_dataset,
    require_disjoint_edge_datasets,
    require_edge_sequence_dataset,
    write_edge_sequence_dataset,
)
from puffer4_edge_artifact_support import native_build_fingerprint


def _dataset(
    *,
    split: str = "train",
    seed: int = 11,
    appearance_seed: int = 41,
    execution_policy: str = "privileged_teacher",
    execution_checkpoint_identity: dict[str, str] | None = None,
    collection_profile: dict[str, float] | None = None,
) -> EdgeSequenceDataset:
    steps, agents = 3, 2
    telemetry = np.zeros((steps, agents, 19), dtype=np.float32)
    telemetry[..., 8] = 1.0
    telemetry[..., 14] = 1.0
    grounding = np.zeros((steps, agents, 4), dtype=np.float32)
    grounding[0, :, :] = (1.0, -0.25, 0.5, 0.2)
    resets = np.zeros((steps, agents), dtype=np.uint8)
    resets[0] = 1
    dones = np.zeros((steps, agents), dtype=np.uint8)
    dones[1, 0] = 1
    resets[2, 0] = 1
    profile = collection_profile
    if profile is None:
        profile = {
            "obstacle_probability": 0.5,
            "camera_randomization": 1.0,
            "layout_diversity": 1.0,
        }
    environment = "flightrl_edge_door"
    fingerprint = native_build_fingerprint(
        Path("/tmp/flightrl-edge-test-build"), environment
    )
    return EdgeSequenceDataset(
        packed_frames=np.zeros((steps, agents, 1536), dtype=np.uint8),
        telemetry=telemetry,
        target_ids=np.zeros((steps, agents), dtype=np.uint8),
        teacher_actions=np.zeros((steps, agents, 4), dtype=np.float32),
        behavior_actions=np.zeros((steps, agents, 4), dtype=np.float32),
        execution_student_mask=np.zeros(agents, dtype=np.uint8),
        grounding=grounding,
        resets=resets,
        dones=dones,
        metadata=edge_dataset_metadata(
            split=split,
            base_seed=seed,
            appearance_seed=appearance_seed,
            steps=steps,
            agents=agents,
            target_ids=(0,),
            environment=environment,
            native_build_fingerprint=fingerprint,
            collection_profile=profile,
            environment_config=canonical_edge_environment_config(
                environment=environment,
                agents=agents,
                base_seed=seed,
                appearance_seed=appearance_seed,
                collection_profile=profile,
            ),
            execution_policy=execution_policy,
            execution_checkpoint_identity=execution_checkpoint_identity,
            execution_student_fraction=(
                0.5 if execution_policy == "dagger_student" else None
            ),
            execution_mix_seed=(seed if execution_policy == "dagger_student" else None),
        ),
    )


def test_edge_sequence_round_trip_preserves_exact_model_abi(tmp_path: Path) -> None:
    dataset = _dataset()
    output = tmp_path / "edge-train.npz"

    write_edge_sequence_dataset(output, dataset)
    loaded = load_edge_sequence_dataset(output)

    assert loaded.shape == (3, 2)
    assert loaded.model_observation(0).shape == (2, 3094)
    assert loaded.model_observation(0)[:, -3:].tolist() == [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert np.array_equal(loaded.resets, dataset.resets)


def test_edge_sequence_requires_terminal_to_next_record_reset() -> None:
    dataset = _dataset()
    dataset.resets[2, 0] = 0

    with pytest.raises(ValueError, match="chronology"):
        require_edge_sequence_dataset(dataset)


def test_edge_sequence_rejects_privileged_or_invalid_values() -> None:
    dataset = _dataset()
    dataset.telemetry[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        require_edge_sequence_dataset(dataset)

    dataset = _dataset()
    dataset.grounding[1, 0, 1] = 0.25
    with pytest.raises(ValueError, match="absent-target"):
        require_edge_sequence_dataset(dataset)

    dataset = _dataset()
    dataset.target_ids[0, 0] = 1
    with pytest.raises(ValueError, match="target"):
        require_edge_sequence_dataset(dataset)


def test_edge_sequence_splits_must_be_seed_and_name_disjoint() -> None:
    train = _dataset()
    selection = _dataset(split="selection", seed=21, appearance_seed=51)
    final = _dataset(split="final", seed=31, appearance_seed=61)

    require_disjoint_edge_datasets(train, selection, final)

    with pytest.raises(ValueError, match="disjoint"):
        require_disjoint_edge_datasets(train, _dataset(split="selection"))
    with pytest.raises(ValueError, match="disjoint"):
        require_disjoint_edge_datasets(train, _dataset(seed=22, appearance_seed=52))
    with pytest.raises(ValueError, match="disjoint"):
        require_disjoint_edge_datasets(
            train,
            _dataset(split="selection", seed=11, appearance_seed=52),
        )
    with pytest.raises(ValueError, match="disjoint"):
        require_disjoint_edge_datasets(
            train,
            _dataset(split="selection", seed=22, appearance_seed=41),
        )
    with pytest.raises(ValueError, match="canonical"):
        require_disjoint_edge_datasets(
            _dataset(split="selection", seed=11, appearance_seed=41),
            _dataset(split="final", seed=21, appearance_seed=51),
        )


def test_edge_sequence_rejects_metadata_tampering() -> None:
    dataset = _dataset()
    tampered = replace(dataset, metadata={**dataset.metadata, "target_ids": [0, 1]})

    with pytest.raises(ValueError, match="door-only"):
        require_edge_sequence_dataset(tampered)


def test_edge_sequence_v4_binds_full_config_hash_and_collection_sources() -> None:
    dataset = _dataset()
    assert dataset.metadata["schema"] == "flightrl.edge_v3.sequence_dataset.v4"
    assert dataset.metadata["environment_config"]["seed"] == 11
    assert set(dataset.metadata["collection_source_identity"]) == {
        "collector", "artifact_paths", "adapter", "dagger", "sequence",
        "collection_evidence", "exporter", "sections", "door_sections", "config",
        "runner", "mission", "native_identity",
    }
    assert dataset.metadata["native_build_fingerprint"] == native_build_fingerprint(
        Path("/tmp/flightrl-edge-test-build"), "flightrl_edge_door"
    )

    dataset.metadata["schema"] = "flightrl.edge_v3.sequence_dataset.v3"
    with pytest.raises(ValueError, match="schema"):
        require_edge_sequence_dataset(dataset)


def test_edge_sequence_rejects_rehashed_noncanonical_config_and_sources() -> None:
    dataset = _dataset()
    dataset.metadata["environment_config"]["max_horizontal_speed_m_s"] = 0.9
    dataset.metadata["environment_config_sha256"] = edge_environment_config_sha256(
        dataset.metadata["environment_config"]
    )
    with pytest.raises(ValueError, match="config is not canonical"):
        require_edge_sequence_dataset(dataset)

    dataset = _dataset()
    dataset.metadata["collection_source_identity"]["adapter"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source identity"):
        require_edge_sequence_dataset(dataset)


def test_edge_sequence_canonically_binds_explicit_collection_overrides() -> None:
    profile = {
        "obstacle_probability": 0.75,
        "camera_randomization": 0.25,
        "layout_diversity": 0.5,
    }
    dataset = _dataset(collection_profile=profile)

    require_edge_sequence_dataset(dataset)
    assert dataset.metadata["collection_profile"] == profile
    assert all(
        dataset.metadata["environment_config"][name] == value
        for name, value in profile.items()
    )

    dataset.metadata["collection_profile"]["obstacle_probability"] = 0.5
    with pytest.raises(ValueError, match="config is not canonical"):
        require_edge_sequence_dataset(dataset)


def test_edge_sequence_binds_dagger_execution_checkpoint_and_fixed_mix() -> None:
    identity = {"path": "/tmp/edge-student.pt", "sha256": "d" * 64}
    dataset = _dataset(
        execution_policy="dagger_student",
        execution_checkpoint_identity=identity,
    )
    dataset.execution_student_mask[:] = fixed_student_mask(
        dataset.metadata,
        dataset.shape[1],
    )

    with pytest.raises(ValueError, match="checkpoint is unavailable"):
        require_edge_sequence_dataset(dataset)

    assert dataset.metadata["execution_checkpoint_identity"] == identity
    assert dataset.metadata["execution_mix"] == {
        "teacher": 0.5,
        "student": 0.5,
        "schedule": "fixed_per_agent_sha256_rank_v1",
        "seed": 11,
    }


def test_edge_sequence_rejects_unbound_or_false_dagger_provenance() -> None:
    with pytest.raises(ValueError, match="checkpoint identity"):
        _dataset(execution_policy="dagger_student")

    with pytest.raises(ValueError, match="teacher.*checkpoint"):
        _dataset(
            execution_checkpoint_identity={
                "path": "/tmp/edge-student.pt",
                "sha256": "d" * 64,
            }
        )

    with pytest.raises(ValueError, match="train split"):
        _dataset(
            split="selection",
            execution_policy="dagger_student",
            execution_checkpoint_identity={
                "path": "/tmp/edge-student.pt",
                "sha256": "d" * 64,
            },
        )
