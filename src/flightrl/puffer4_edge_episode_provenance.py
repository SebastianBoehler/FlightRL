from __future__ import annotations

import numpy as np


def require_edge_episode_provenance(
    episode_ids: np.ndarray,
    scene_group_ids: np.ndarray,
    resets: np.ndarray,
    visible: np.ndarray,
) -> None:
    reset = resets.astype(bool)
    starts = episode_ids[reset]
    if np.unique(starts).size != starts.size:
        raise ValueError("edge dataset episode IDs must be unique per segment")
    if np.any(scene_group_ids > 127):
        raise ValueError("edge dataset scene group IDs use reserved bits")
    initial_outside_fov = (scene_group_ids[reset] & 64) != 0
    if np.any(initial_outside_fov != (visible[reset] <= 0.5)):
        raise ValueError(
            "edge dataset initial outside-FOV group disagrees with reset visibility"
        )
    if episode_ids.shape[0] <= 1:
        return
    continuation = ~reset[1:]
    episode_changed = episode_ids[1:] != episode_ids[:-1]
    group_changed = scene_group_ids[1:] != scene_group_ids[:-1]
    if np.any(episode_changed != reset[1:]):
        raise ValueError("edge dataset episode IDs do not follow reset boundaries")
    if np.any(group_changed & continuation):
        raise ValueError("edge dataset scene group changed within an episode")
