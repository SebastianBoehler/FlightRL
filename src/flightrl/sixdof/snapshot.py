from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SixDofSnapshot:
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    body_rates: np.ndarray
    target_position: np.ndarray
    target_yaw: np.ndarray
    ranges_m: np.ndarray
