from __future__ import annotations

import numpy as np

from flightrl.sixdof.curriculum import ResetProfile, sample_reset
from flightrl.sixdof.geometry import BoxRoom
from flightrl.sixdof.validation import require_finite_real, require_positive_int


BOX_ROOM_RESET_CLEARANCE_M = 0.08
SEMANTIC_RESET_CLEARANCE_M = 0.35


def sample_collision_free_reset(
    profile: ResetProfile,
    rng: np.random.Generator,
    count: int,
    room: BoxRoom,
    *,
    clearance_m: float,
) -> tuple[np.ndarray, ...]:
    count = require_positive_int(count, "reset count")
    clearance_m = require_finite_real(
        clearance_m,
        "reset clearance",
        minimum=0.0,
        strictly_greater=True,
    )
    accepted = [[] for _ in range(6)]
    remaining = count
    for _attempt in range(64):
        batch = max(8, 2 * remaining)
        sampled = sample_reset(profile, rng, batch, room)
        valid = (
            room.contains(sampled[0], margin=clearance_m)
            & room.contains(sampled[4], margin=clearance_m)
        )
        selected = np.flatnonzero(valid)[:remaining]
        for output, values in zip(accepted, sampled, strict=True):
            output.extend(values[selected])
        remaining = count - len(accepted[0])
        if remaining == 0:
            return tuple(np.asarray(values) for values in accepted)
    raise RuntimeError(
        f"could not sample {count} reset poses and targets with "
        f"{clearance_m:.2f} m obstacle clearance"
    )
