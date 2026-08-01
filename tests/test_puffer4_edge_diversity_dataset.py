from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from flightrl.puffer4_edge_sequence import (
    load_edge_sequence_dataset,
    require_edge_sequence_dataset,
    write_edge_sequence_dataset,
)
from puffer4_edge_training_support import training_dataset


def test_schema_v5_round_trip_binds_episode_and_scene_groups(tmp_path) -> None:
    dataset = training_dataset("train", 11)
    output = tmp_path / "sequence-v5.npz"

    write_edge_sequence_dataset(output, dataset)
    loaded = load_edge_sequence_dataset(output)

    assert loaded.metadata["schema"] == "flightrl.edge_v3.sequence_dataset.v5"
    np.testing.assert_array_equal(loaded.episode_ids, dataset.episode_ids)
    np.testing.assert_array_equal(loaded.scene_group_ids, dataset.scene_group_ids)


def test_sequence_rejects_reused_episode_id_or_group_change_mid_segment() -> None:
    dataset = training_dataset("train", 11)
    episode_ids = dataset.episode_ids.copy()
    episode_ids[0, 1] = episode_ids[0, 0]

    with pytest.raises(ValueError, match="episode IDs"):
        require_edge_sequence_dataset(replace(dataset, episode_ids=episode_ids))

    groups = dataset.scene_group_ids.copy()
    groups[1, 0] ^= 1
    with pytest.raises(ValueError, match="scene group"):
        require_edge_sequence_dataset(replace(dataset, scene_group_ids=groups))


def test_reset_scene_group_must_match_rendered_visibility() -> None:
    dataset = training_dataset("train", 11)
    groups = dataset.scene_group_ids.copy()
    groups[:, 0] ^= 64

    with pytest.raises(ValueError, match="initial outside-FOV"):
        require_edge_sequence_dataset(replace(dataset, scene_group_ids=groups))
