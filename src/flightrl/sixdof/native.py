from __future__ import annotations

import numpy as np

from flightrl import _binding


def native_step(
    position: np.ndarray,
    velocity: np.ndarray,
    quaternion: np.ndarray,
    body_rates: np.ndarray,
    ranges_m: np.ndarray,
    actions: np.ndarray,
    dt: float,
) -> None:
    _binding.sixdof_step(
        _float32(position),
        _float32(velocity),
        _float32(quaternion),
        _float32(body_rates),
        _float32(ranges_m),
        _float32(actions),
        float(dt),
    )


def _float32(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.float32 and values.flags.c_contiguous:
        return values
    raise ValueError("native 6-DoF arrays must be C-contiguous float32")
