from __future__ import annotations

import numpy as np


def latched_target(rows: list[dict[str, float]], fallback: tuple[float, float, float] | np.ndarray, mode: str) -> np.ndarray:
    base = np.asarray(fallback, dtype=np.float32)
    if mode == "fixed_origin" or not rows:
        return base
    if mode != "current_pose":
        raise ValueError(f"unknown target mode {mode!r}")
    first = rows[0]
    return np.asarray([value(first, "stateEstimate.x"), value(first, "stateEstimate.y"), base[2]], dtype=np.float32)


def value(row: dict[str, float], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
