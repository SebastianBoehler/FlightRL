from __future__ import annotations

import numpy as np

from .range_contract import RANGE_EXPLORATION_OBSERVATION_DIM, RANGE_MAP_SHAPE


def build_range_exploration_observation(
    exploration_map: np.ndarray,
    horizontal_ranges: np.ndarray,
    range_validity: np.ndarray,
    previous_applied_action: np.ndarray,
) -> np.ndarray:
    map_value = np.asarray(exploration_map, dtype=np.float32)
    ranges = np.asarray(horizontal_ranges, dtype=np.float32)
    validity = np.asarray(range_validity, dtype=np.float32)
    action = np.asarray(previous_applied_action, dtype=np.float32)
    if map_value.shape != RANGE_MAP_SHAPE:
        raise ValueError(f"range exploration map shape must be {RANGE_MAP_SHAPE}")
    if ranges.shape != (4,):
        raise ValueError("horizontal ranges must have shape (4,)")
    if validity.shape != (4,):
        raise ValueError("range validity must have shape (4,)")
    if action.shape != (2,):
        raise ValueError("previous applied action must have shape (2,)")
    if not all(np.isfinite(value).all() for value in (map_value, ranges, validity, action)):
        raise ValueError("range exploration observation values must be finite")
    if np.any((map_value < 0.0) | (map_value > 1.0)):
        raise ValueError("range exploration map values must be in [0, 1]")
    if np.any((ranges < 0.0) | (ranges > 1.0)):
        raise ValueError("horizontal ranges must be normalized to [0, 1]")
    if np.any((validity != 0.0) & (validity != 1.0)):
        raise ValueError("range validity values must be binary")
    if not 0.0 <= float(action[0]) <= 1.0 or not -1.0 <= float(action[1]) <= 1.0:
        raise ValueError("previous applied action violates normalized bounds")
    observation = np.concatenate(
        (map_value.reshape(-1), ranges, validity, action),
        dtype=np.float32,
    )
    if observation.shape != (RANGE_EXPLORATION_OBSERVATION_DIM,):
        raise RuntimeError("range exploration observation layout is inconsistent")
    return observation
