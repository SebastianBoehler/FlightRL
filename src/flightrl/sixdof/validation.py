from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral, Real

import numpy as np


_BOOL_TYPES = (bool, np.bool_)


def require_bool(value: object, name: str) -> bool:
    if not isinstance(value, _BOOL_TYPES):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def require_positive_int(value: object, name: str) -> int:
    if isinstance(value, _BOOL_TYPES) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def require_finite_real(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    strictly_greater: bool = False,
) -> float:
    if isinstance(value, _BOOL_TYPES) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        invalid = result <= minimum if strictly_greater else result < minimum
        if invalid:
            relation = "greater than" if strictly_greater else "at least"
            raise ValueError(f"{name} must be {relation} {minimum:g}")
    return result


def require_real_tuple(
    value: object,
    name: str,
    length: int,
    *,
    minimum: float | None = None,
    strictly_greater: bool = False,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must contain exactly {length} real numbers")
    if len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} real numbers")
    return tuple(
        require_finite_real(
            item,
            f"{name}[{index}]",
            minimum=minimum,
            strictly_greater=strictly_greater,
        )
        for index, item in enumerate(value)
    )


def require_interval(
    value: object,
    name: str,
    *,
    minimum: float = 0.0,
    strictly_positive: bool = False,
) -> tuple[float, float]:
    low, high = require_real_tuple(
        value,
        name,
        2,
        minimum=minimum,
        strictly_greater=strictly_positive,
    )
    if low > high:
        raise ValueError(f"{name} must be sorted low-to-high")
    return low, high


def require_choice(value: object, name: str, choices: Mapping | set | tuple) -> str:
    if not isinstance(value, str) or value not in choices:
        expected = ", ".join(sorted(str(choice) for choice in choices))
        raise ValueError(f"unknown {name} {value!r}; expected one of {expected}")
    return value


def action_batch(actions: object, num_envs: int, action_dim: int) -> np.ndarray:
    raw = np.asarray(actions)
    expected = (num_envs, action_dim)
    if raw.shape != expected:
        raise ValueError(f"action batch must have shape {expected}, got {raw.shape}")
    if raw.dtype.kind not in "fiu" or raw.dtype.kind == "b":
        raise TypeError("action batch must contain real numeric values")
    if not np.all(np.isfinite(raw)):
        raise ValueError("action batch must contain only finite values")
    return np.clip(raw.astype(np.float32), -1.0, 1.0)


def reset_mask(values: object, num_envs: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (num_envs,):
        raise ValueError(
            f"reset mask must have shape {(num_envs,)}, got {raw.shape}"
        )
    if raw.dtype.kind == "b":
        return raw.astype(bool, copy=False)
    if raw.dtype.kind in "iu":
        if np.all((raw == 0) | (raw == 1)):
            return raw.astype(bool)
        raise ValueError("reset mask integer values must be 0 or 1")
    raise TypeError("reset mask must contain only booleans or integer 0/1 values")


def task_id_batch(
    task_indices: object,
    tasks: Sequence[str],
    *,
    num_envs: int,
    task_ids: Mapping[str, int],
) -> np.ndarray:
    if not tasks:
        raise ValueError("tasks cannot be empty")
    try:
        ids = np.asarray([task_ids[name] for name in tasks], dtype=np.int32)
    except (KeyError, TypeError) as exc:
        raise ValueError("tasks contain an unknown 6-DoF task") from exc
    indices = np.asarray(task_indices)
    if indices.shape != (num_envs,):
        raise ValueError("task indices must match environment count")
    if indices.dtype.kind not in "iu" or indices.dtype.kind == "b":
        raise TypeError("task indices must be integers")
    if np.any(indices < 0) or np.any(indices >= len(ids)):
        raise ValueError("task indices are outside the supplied tasks")
    return ids[indices.astype(np.int64)]


def finite_batch(values: object, name: str, num_envs: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (num_envs,):
        raise ValueError(f"{name} must match environment count")
    if raw.dtype.kind not in "fiu" or raw.dtype.kind == "b":
        raise TypeError(f"{name} must contain real numeric values")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain only finite values")
    return raw.astype(np.float32)
