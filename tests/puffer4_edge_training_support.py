from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flightrl.puffer4_edge_collection_evidence import (
    canonical_edge_environment_config,
)
from flightrl.puffer4_edge_sequence import (
    EdgeSequenceDataset,
    edge_dataset_metadata,
)
from puffer4_edge_artifact_support import native_build_fingerprint


@pytest.fixture
def allow_tiny_training_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    import flightrl.puffer4_edge_training as edge_training

    monkeypatch.setattr(
        edge_training,
        "require_edge_training_coverage",
        lambda train, selection: {
            "train": {"segments": int(train.resets.sum())},
            "selection": {"segments": int(selection.resets.sum())},
        },
    )


def training_dataset(split: str, seed: int) -> EdgeSequenceDataset:
    steps, agents = 8, 2
    telemetry = np.zeros((steps, agents, 19), dtype=np.float32)
    telemetry[..., 8] = 1.0
    telemetry[..., 14] = 1.0
    actions = np.zeros((steps, agents, 4), dtype=np.float32)
    bright = np.indices((steps, agents)).sum(axis=0) % 2 == 1
    actions[..., 0][bright] = 0.8
    actions[..., 3][~bright] = 1.0
    grounding = np.zeros((steps, agents, 4), dtype=np.float32)
    grounding[bright] = (1.0, 0.25, -0.25, 0.4)
    resets = np.zeros((steps, agents), dtype=np.uint8)
    resets[0] = 1
    episode_ids = np.broadcast_to(
        np.arange(agents, dtype=np.uint64), (steps, agents)
    ).copy()
    scene_groups = np.broadcast_to(
        np.where(bright[0], 0, 64).astype(np.uint8), (steps, agents)
    ).copy()
    frames = np.zeros((steps, agents, 1536), dtype=np.uint8)
    frames[bright] = 255
    return EdgeSequenceDataset(
        packed_frames=frames,
        telemetry=telemetry,
        target_ids=np.zeros((steps, agents), dtype=np.uint8),
        teacher_actions=actions,
        behavior_actions=actions.copy(),
        execution_student_mask=np.zeros(agents, dtype=np.uint8),
        grounding=grounding,
        episode_ids=episode_ids,
        scene_group_ids=scene_groups,
        resets=resets,
        dones=np.zeros((steps, agents), dtype=np.uint8),
        metadata=training_metadata(split, seed, steps, agents),
    )


def training_metadata(split: str, seed: int, steps: int, agents: int) -> dict:
    profile = {
        "obstacle_probability": 0.5,
        "camera_randomization": 1.0,
        "layout_diversity": 1.0,
    }
    environment = "edge-door-test"
    return edge_dataset_metadata(
        split=split,
        base_seed=seed,
        appearance_seed=seed + 100,
        steps=steps,
        agents=agents,
        target_ids=(0,),
        environment=environment,
        native_build_fingerprint=native_build_fingerprint(
            Path("/tmp/flightrl-edge-training-test"), environment
        ),
        collection_profile=profile,
        environment_config=canonical_edge_environment_config(
            environment=environment,
            agents=agents,
            base_seed=seed,
            appearance_seed=seed + 100,
            collection_profile=profile,
        ),
    )
