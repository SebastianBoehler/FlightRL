from __future__ import annotations

import numpy as np

from flightrl.navigation import Bounds3D
from flightrl.navigation.spatial_memory import (
    MAP_CHANNELS,
    EgocentricSpatialMemory,
    SpatialMemoryConfig,
)


def test_spatial_memory_tracks_visited_rays_and_semantic_evidence() -> None:
    config = SpatialMemoryConfig(cell_size_m=0.25, local_size=16)
    memory = EgocentricSpatialMemory(
        Bounds3D((-2.0, -2.0, 0.0), (2.0, 2.0, 2.5)),
        config,
    )

    assert memory.update_pose(np.asarray((0.0, 0.0))) > 0
    assert memory.update_pose(np.asarray((0.0, 0.0))) == 0
    memory.observe_rays(
        np.asarray((0.0, 0.0)),
        0.0,
        np.asarray((0.0,)),
        np.asarray((1.0,)),
        max_range_m=2.0,
    )
    memory.observe_semantic(
        np.asarray((0.0, 0.0)),
        0.0,
        0.0,
        1.0,
        0.8,
    )
    local = memory.local_map(np.asarray((0.0, 0.0)), 0.0)

    assert local.shape == config.shape
    assert local[MAP_CHANNELS.index("visited")].sum() > 0
    assert local[MAP_CHANNELS.index("observed_free")].sum() > 0
    assert local[MAP_CHANNELS.index("target_evidence")].max() == np.float32(0.8)
    assert local[MAP_CHANNELS.index("agent"), 8, 8] == 1.0


def test_spatial_memory_can_replace_stale_semantic_evidence() -> None:
    memory = EgocentricSpatialMemory(
        Bounds3D((-2.0, -2.0, 0.0), (2.0, 2.0, 2.5)),
        SpatialMemoryConfig(cell_size_m=0.25, local_size=16),
    )
    origin = np.asarray((0.0, 0.0))
    memory.observe_semantic(origin, 0.0, 0.0, 1.0, 0.8)
    memory.observe_semantic(
        origin,
        0.0,
        np.pi,
        1.0,
        0.6,
        replace=True,
    )

    target = memory.grid[MAP_CHANNELS.index("target_evidence")]
    assert target.max() == np.float32(0.6)
    assert np.count_nonzero(target) > 0
