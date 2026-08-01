from __future__ import annotations

import numpy as np

from scripts.build_puffer_edge_dataset import collect_dataset
from test_build_puffer_edge_dataset import (
    _FakeTorchPuffer,
    _FakeVec,
    _metadata,
    _native_observation,
)


def test_collector_assigns_unique_episode_ids_and_native_scene_groups() -> None:
    observations = _native_observation(2)
    observations[:, -1] = 109.0
    vec = _FakeVec(observations, [[1, 0], [0, 1], [0, 0]])

    dataset = collect_dataset(
        {},
        _FakeTorchPuffer(vec),
        steps=3,
        agents=2,
        metadata=_metadata(steps=3, agents=2),
    )

    np.testing.assert_array_equal(
        dataset.episode_ids,
        np.asarray(((0, 1), (2, 1), (2, 3)), dtype=np.uint64),
    )
    np.testing.assert_array_equal(dataset.scene_group_ids, 109)
