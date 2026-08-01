from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def allocate_edge_collection_arrays(
    steps: int,
    agents: int,
) -> dict[str, np.ndarray]:
    prefix = (steps, agents)
    return {
        "packed_frames": np.empty(prefix + (1536,), dtype=np.uint8),
        "telemetry": np.empty(prefix + (19,), dtype=np.float32),
        "target_ids": np.empty(prefix, dtype=np.uint8),
        "teacher_actions": np.empty(prefix + (4,), dtype=np.float32),
        "behavior_actions": np.empty(prefix + (4,), dtype=np.float32),
        "execution_student_mask": np.empty(agents, dtype=np.uint8),
        "grounding": np.empty(prefix + (4,), dtype=np.float32),
        "episode_ids": np.empty(prefix, dtype=np.uint64),
        "scene_group_ids": np.empty(prefix, dtype=np.uint8),
        "resets": np.empty(prefix, dtype=np.uint8),
        "dones": np.empty(prefix, dtype=np.uint8),
    }


@dataclass(slots=True)
class EdgeEpisodeIdTracker:
    active: np.ndarray
    next_id: int = 0

    @classmethod
    def create(cls, agents: int) -> "EdgeEpisodeIdTracker":
        return cls(np.zeros(agents, dtype=np.uint64))

    def assign(self, reset: np.ndarray) -> np.ndarray:
        indices = np.flatnonzero(reset)
        end = self.next_id + len(indices)
        self.active[indices] = np.arange(
            self.next_id,
            end,
            dtype=np.uint64,
        )
        self.next_id = end
        return self.active.copy()
