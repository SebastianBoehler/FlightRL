from __future__ import annotations

import numpy as np
import pytest

from flightrl.puffer4_edge_coverage import (
    edge_realized_coverage,
    require_edge_training_coverage,
)
from flightrl.puffer4_edge_sequence import EdgeSequenceDataset
from puffer4_edge_training_support import training_metadata


def _coverage_dataset(split: str) -> EdgeSequenceDataset:
    agents = 256 if split == "train" else 128
    steps = 8 if split == "train" else 2
    prefix = (steps, agents)
    resets = np.zeros(prefix, dtype=np.uint8)
    resets[:2] = 1
    dones = np.zeros(prefix, dtype=np.uint8)
    dones[0] = 1
    episode_ids = np.empty(prefix, dtype=np.uint64)
    episode_ids[0] = np.arange(agents, dtype=np.uint64)
    episode_ids[1:] = np.arange(agents, 2 * agents, dtype=np.uint64)
    segment = np.arange(2 * agents, dtype=np.uint16)
    layout = segment % 4
    door_face = (segment // 4) % 4
    low_light = segment % 10 == 0
    obstacle = segment % 2 == 0
    outside_fov = segment % 4 < 2
    groups = (
        layout
        | (door_face << 2)
        | (low_light.astype(np.uint16) << 4)
        | (obstacle.astype(np.uint16) << 5)
        | (outside_fov.astype(np.uint16) << 6)
    ).astype(np.uint8)
    scene_groups = np.empty(prefix, dtype=np.uint8)
    scene_groups[0] = groups[:agents]
    scene_groups[1:] = groups[agents:]
    telemetry = np.zeros(prefix + (19,), dtype=np.float32)
    telemetry[..., 8] = 1.0
    telemetry[..., 14] = 1.0
    actions = np.zeros(prefix + (4,), dtype=np.float32)
    actions[1::2, :, 0] = 1.0
    grounding = np.zeros(prefix + (4,), dtype=np.float32)
    visible = (scene_groups & 64) == 0
    grounding[visible] = (1.0, 0.0, 0.0, 0.4)
    return EdgeSequenceDataset(
        packed_frames=np.zeros(prefix + (1536,), dtype=np.uint8),
        telemetry=telemetry,
        target_ids=np.zeros(prefix, dtype=np.uint8),
        teacher_actions=actions,
        behavior_actions=actions.copy(),
        execution_student_mask=np.zeros(agents, dtype=np.uint8),
        grounding=grounding,
        episode_ids=episode_ids,
        scene_group_ids=scene_groups,
        resets=resets,
        dones=dones,
        metadata=training_metadata(split, 11 if split == "train" else 21, steps, agents),
    )


def test_realized_coverage_accepts_canonical_256_128_lane() -> None:
    train = _coverage_dataset("train")
    selection = _coverage_dataset("selection")

    report = require_edge_training_coverage(train, selection)

    assert report["train"]["segments"] == 512
    assert report["train"]["critical_events"] == 2048
    assert report["selection"]["segments"] == 256
    assert report["selection"]["initial_visible"] == 128
    assert report["selection"]["initial_outside_fov"] == 128


def test_coverage_error_names_every_deficient_field() -> None:
    train = _coverage_dataset("train")
    train.scene_group_ids[:] = 0
    train.teacher_actions[:] = 0.0
    train.grounding[:] = (1.0, 0.0, 0.0, 0.4)

    with pytest.raises(ValueError) as caught:
        require_edge_training_coverage(train, _coverage_dataset("selection"))

    message = str(caught.value)
    assert "critical_events" in message
    assert "initial_outside_fov" in message
    assert "layout_family_3" in message
    assert "door_face_3" in message
    assert "low_light" in message
    assert "obstacle" in message
    assert edge_realized_coverage(train)["segments"] == 512
