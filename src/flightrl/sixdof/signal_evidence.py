from __future__ import annotations

from flightrl.evidence_values import exact_nonnegative_int, finite_number


RANGE_SIGNALS = (
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "range.zrange",
)
REPLAY_STATE_SIGNALS = (
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
)
NATIVE_STATE_SIGNALS = (
    *REPLAY_STATE_SIGNALS,
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.vz",
    "stabilizer.yaw",
)


def worst_complete_rmse(
    signals: object,
    required: tuple[str, ...],
    *,
    detailed: bool = False,
) -> float | None:
    if not isinstance(signals, dict) or not set(required).issubset(signals):
        return None
    values = [signal_rmse(signals[name], detailed=detailed) for name in required]
    return max(values) if all(value is not None for value in values) else None


def signal_rmse(metrics: object, *, detailed: bool = False) -> float | None:
    if not isinstance(metrics, dict):
        return None
    samples = exact_nonnegative_int(metrics.get("samples"))
    rmse = finite_nonnegative(metrics.get("rmse"))
    if samples is None or samples < 2 or rmse is None:
        return None
    if detailed:
        mae = finite_nonnegative(metrics.get("mae"))
        maximum = finite_nonnegative(metrics.get("max_abs"))
        if mae is None or maximum is None or mae > rmse or rmse > maximum:
            return None
    return rmse


def finite_nonnegative(value: object) -> float | None:
    parsed = finite_number(value)
    return parsed if parsed is not None and parsed >= 0.0 else None
