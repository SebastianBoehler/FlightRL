from __future__ import annotations

import numpy as np

from .env import ACTION_DIM


OBSERVATION_MODES = ("base", "history1")


def augment_observation(current: np.ndarray, previous: np.ndarray, previous_action: np.ndarray, mode: str) -> np.ndarray:
    if mode == "base":
        return current.astype(np.float32)
    if mode == "history1":
        delta = current - previous
        return np.concatenate([current, delta, previous_action], axis=1).astype(np.float32)
    raise ValueError(f"unknown observation mode {mode!r}")


def observation_dim(base_dim: int, mode: str) -> int:
    if mode == "base":
        return int(base_dim)
    if mode == "history1":
        return 2 * int(base_dim) + ACTION_DIM
    raise ValueError(f"unknown observation mode {mode!r}")
