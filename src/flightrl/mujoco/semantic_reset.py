from __future__ import annotations

import numpy as np

from flightrl.sixdof.curriculum import ResetProfile, sample_reset
from flightrl.sixdof.geometry import BoxRoom


SEMANTIC_RESET_CLEARANCE_M = 0.35


def sample_semantic_reset(
    profile: ResetProfile,
    rng: np.random.Generator,
    count: int,
    room: BoxRoom,
) -> tuple[np.ndarray, ...]:
    accepted = [[] for _ in range(6)]
    remaining = count
    for _attempt in range(64):
        batch = max(8, 2 * remaining)
        sampled = sample_reset(profile, rng, batch, room)
        valid = room.contains(
            sampled[0],
            margin=SEMANTIC_RESET_CLEARANCE_M,
        )
        selected = np.flatnonzero(valid)[:remaining]
        for output, values in zip(accepted, sampled, strict=True):
            output.extend(values[selected])
        remaining = count - len(accepted[0])
        if remaining == 0:
            return tuple(np.asarray(values) for values in accepted)
    raise RuntimeError(
        f"could not sample {count} semantic reset poses with "
        f"{SEMANTIC_RESET_CLEARANCE_M:.2f} m clearance"
    )
