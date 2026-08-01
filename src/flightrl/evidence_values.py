from __future__ import annotations

import math
from numbers import Real


def finite_number(value: object) -> float | None:
    """Return a finite real evidence value without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def exact_nonnegative_int(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    return None


def exact_true(value: object) -> bool:
    return value is True


def failure_strings(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, str) and item for item in value):
        return None
    return list(value)
