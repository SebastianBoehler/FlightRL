from __future__ import annotations

from math import isfinite
from typing import Any


def fit_motor_curve(
    rows: list[dict[str, float | int | None]],
    *,
    min_powers: int,
    min_r2: float,
    min_valid_rpm: float = 0.0,
    max_dropout_ratio: float = 0.0,
) -> dict[str, Any]:
    raw_points = sorted(
        [(float(row["power"]), float(row["rpm"])) for row in rows],
        key=lambda item: item[0],
    )
    points, dropped = filter_motor_dropouts(
        raw_points,
        min_valid_rpm=min_valid_rpm,
        max_dropout_ratio=max_dropout_ratio,
    )
    powers = {power for power, _rpm in points}
    slope, intercept, r2 = linear_fit(points)
    failures = []
    if len(powers) < min_powers:
        failures.append("power_coverage")
    if not any(rpm > 0 for _power, rpm in points):
        failures.append("rpm_signal")
    if slope <= 0:
        failures.append("slope")
    if r2 is None or r2 < min_r2:
        failures.append("r2")
    if not monotonic(points):
        failures.append("monotonicity")
    return {
        "passed": not failures,
        "failures": failures,
        "samples": len(raw_points),
        "filtered_samples": len(points),
        "dropped_samples": len(dropped),
        "dropped": dropped,
        "power_min": min(powers) if powers else None,
        "power_max": max(powers) if powers else None,
        "rpm_min": min([rpm for _power, rpm in points], default=None),
        "rpm_max": max([rpm for _power, rpm in points], default=None),
        "slope_rpm_per_power": slope,
        "intercept_rpm": intercept,
        "r2": r2,
    }


def filter_motor_dropouts(
    points: list[tuple[float, float]],
    *,
    min_valid_rpm: float,
    max_dropout_ratio: float,
) -> tuple[list[tuple[float, float]], list[dict[str, float | str]]]:
    kept: list[tuple[float, float]] = []
    dropped: list[dict[str, float | str]] = []
    for power, rpm in points:
        reason = None
        if min_valid_rpm > 0.0 and rpm < min_valid_rpm:
            reason = "rpm_below_min"
        elif max_dropout_ratio > 0.0 and kept and rpm < kept[-1][1] * max_dropout_ratio:
            reason = "rpm_dropout"
        if reason:
            dropped.append({"power": power, "rpm": rpm, "reason": reason})
            continue
        kept.append((power, rpm))
    return kept, dropped


def linear_fit(
    points: list[tuple[float, float]],
) -> tuple[float, float, float | None]:
    if len(points) < 2:
        return 0.0, 0.0, None
    xs, ys = zip(*points)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
        if denom
        else 0.0
    )
    intercept = mean_y - slope * mean_x
    total = sum((y - mean_y) ** 2 for y in ys)
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    r2 = 1.0 - residual / total if total else None
    return slope, intercept, r2


def simulator_priors(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    passed = {motor: record for motor, record in records.items() if record["passed"]}
    if not passed:
        return {"present": False}
    mean_slope = sum(
        record["slope_rpm_per_power"] for record in passed.values()
    ) / len(passed)
    return {
        "present": True,
        "mean_slope_rpm_per_power": mean_slope,
        "relative_motor_gains": {
            motor: record["slope_rpm_per_power"] / mean_slope
            for motor, record in passed.items()
        },
    }


def monotonic(points: list[tuple[float, float]]) -> bool:
    return all(
        next_rpm >= rpm
        for (_power, rpm), (_next_power, next_rpm) in zip(points, points[1:])
    )


def slope_imbalance(slopes: list[float]) -> float | None:
    if len(slopes) != 4 or min(slopes) <= 0:
        return None
    mean = sum(slopes) / len(slopes)
    return max(abs(slope - mean) / mean for slope in slopes)


def validate_fit_thresholds(
    *,
    min_powers: int,
    min_r2: float,
    max_gain_imbalance: float,
    min_valid_rpm: float,
    max_dropout_ratio: float,
) -> None:
    if type(min_powers) is not int or min_powers < 2:
        raise ValueError("min_powers must be an integer >= 2")
    values = (min_r2, max_gain_imbalance, min_valid_rpm, max_dropout_ratio)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        for value in values
    ):
        raise ValueError("motor calibration thresholds must be finite numbers")
    if not 0.0 <= min_r2 <= 1.0:
        raise ValueError("min_r2 must be in [0, 1]")
    if not 0.0 <= max_gain_imbalance <= 1.0:
        raise ValueError("max_gain_imbalance must be in [0, 1]")
    if min_valid_rpm < 0.0:
        raise ValueError("min_valid_rpm must be nonnegative")
    if not 0.0 <= max_dropout_ratio <= 1.0:
        raise ValueError("max_dropout_ratio must be in [0, 1]")
