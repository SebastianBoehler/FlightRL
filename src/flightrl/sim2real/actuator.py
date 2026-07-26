from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def summarize_motor_bench(path: Path | None, *, min_powers: int) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "passed": False, "motors": {}, "failures": ["missing"]}
    rows = load_motor_rows(path)
    motors = group_motor_rows(rows)
    compact = {str(motor): motor_stats(values) for motor, values in sorted(motors.items())}
    failures = []
    if set(motors) != {1, 2, 3, 4}:
        failures.append("motor_coverage")
    if any(len({row["power"] for row in values}) < min_powers for values in motors.values()):
        failures.append("power_coverage")
    if any(not any(row["rpm"] > 0 for row in values) for values in motors.values()):
        failures.append("rpm_signal")
    if any(not any(row["vbat"] is not None for row in values) for values in motors.values()):
        failures.append("battery_signal")
    return {"present": True, "path": str(path), "passed": not failures, "motors": compact, "failures": failures}


def fit_motor_calibration(
    path: Path,
    *,
    min_powers: int = 3,
    min_r2: float = 0.9,
    max_gain_imbalance: float = 0.25,
    min_valid_rpm: float = 0.0,
    max_dropout_ratio: float = 0.0,
) -> dict[str, Any]:
    rows = load_motor_rows(path)
    motors = group_motor_rows(rows)
    records = {
        str(motor): fit_motor_curve(
            values,
            min_powers=min_powers,
            min_r2=min_r2,
            min_valid_rpm=min_valid_rpm,
            max_dropout_ratio=max_dropout_ratio,
        )
        for motor, values in sorted(motors.items())
    }
    slopes = [record["slope_rpm_per_power"] for record in records.values() if record["passed"]]
    gain_imbalance = slope_imbalance(slopes)
    dropped_samples = sum(record["dropped_samples"] for record in records.values())
    failures = []
    if set(motors) != {1, 2, 3, 4}:
        failures.append("motor_coverage")
    for motor, record in records.items():
        failures.extend(f"m{motor}_{failure}" for failure in record["failures"])
    if gain_imbalance is None:
        failures.append("gain_imbalance_missing")
    elif gain_imbalance > max_gain_imbalance:
        failures.append("gain_imbalance")
    return {
        "input": str(path),
        "summary": {
            "passed": not failures,
            "failures": sorted(dict.fromkeys(failures)),
            "motor_count": len(motors),
            "gain_imbalance": gain_imbalance,
            "min_r2": min_r2,
            "max_gain_imbalance": max_gain_imbalance,
            "dropped_samples": dropped_samples,
            "warnings": ["rpm_dropouts_filtered"] if dropped_samples else [],
            "vbat": vbat_range(rows),
        },
        "motors": records,
        "simulator_priors": simulator_priors(records),
        "safety": "RPM calibration only; thrust/torque still require a thrust stand or replay fitting.",
    }


def load_motor_rows(path: Path) -> list[dict[str, float | int | None]]:
    rows = []
    for row in csv.DictReader(path.open()):
        motor = as_int(row.get("motor"))
        power = as_float(row.get("power"))
        rpm = as_float(row.get("rpm"))
        if motor is None or power is None or rpm is None:
            continue
        rows.append({"motor": motor, "power": power, "rpm": rpm, "vbat": as_float(row.get("vbat"))})
    return rows


def group_motor_rows(rows: list[dict[str, float | int | None]]) -> dict[int, list[dict[str, float | int | None]]]:
    grouped: dict[int, list[dict[str, float | int | None]]] = {}
    for row in rows:
        grouped.setdefault(int(row["motor"]), []).append(row)
    return grouped


def fit_motor_curve(
    rows: list[dict[str, float | int | None]],
    *,
    min_powers: int,
    min_r2: float,
    min_valid_rpm: float = 0.0,
    max_dropout_ratio: float = 0.0,
) -> dict[str, Any]:
    raw_points = sorted([(float(row["power"]), float(row["rpm"])) for row in rows], key=lambda item: item[0])
    points, dropped = filter_motor_dropouts(raw_points, min_valid_rpm=min_valid_rpm, max_dropout_ratio=max_dropout_ratio)
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


def linear_fit(points: list[tuple[float, float]]) -> tuple[float, float, float | None]:
    if len(points) < 2:
        return 0.0, 0.0, None
    xs, ys = zip(*points)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denom if denom else 0.0
    intercept = mean_y - slope * mean_x
    total = sum((y - mean_y) ** 2 for y in ys)
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    r2 = 1.0 - residual / total if total else None
    return slope, intercept, r2


def simulator_priors(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    passed = {motor: record for motor, record in records.items() if record["passed"]}
    if not passed:
        return {"present": False}
    mean_slope = sum(record["slope_rpm_per_power"] for record in passed.values()) / len(passed)
    return {
        "present": True,
        "mean_slope_rpm_per_power": mean_slope,
        "relative_motor_gains": {motor: record["slope_rpm_per_power"] / mean_slope for motor, record in passed.items()},
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Motor Bench Calibration",
        "",
        f"- Input: `{report['input']}`",
        f"- Passed: `{summary['passed']}`",
        f"- Failures: `{', '.join(summary['failures']) or 'none'}`",
        f"- Gain imbalance: `{summary['gain_imbalance']}`",
        f"- Warnings: `{', '.join(summary.get('warnings', [])) or 'none'}`",
        f"- Dropped samples: `{summary.get('dropped_samples', 0)}`",
        "",
        "| motor | passed | filtered/total | dropped | slope rpm/power | intercept rpm | r2 | rpm min | rpm max | failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for motor, record in report["motors"].items():
        lines.append(
            f"| {motor} | {record['passed']} | {record.get('filtered_samples', record['samples'])}/{record['samples']} | "
            f"{record.get('dropped_samples', 0)} | {_fmt(record['slope_rpm_per_power'])} | {_fmt(record['intercept_rpm'])} | "
            f"{_fmt(record['r2'])} | {_fmt(record['rpm_min'])} | {_fmt(record['rpm_max'])} | {', '.join(record['failures']) or 'none'} |"
        )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def motor_stats(rows: list[dict[str, float | int | None]]) -> dict[str, Any]:
    rpms = [float(row["rpm"]) for row in rows]
    vbats = [row["vbat"] for row in rows if row["vbat"] is not None]
    return {"powers": sorted({row["power"] for row in rows}), "rpm_min": min(rpms) if rpms else None, "rpm_max": max(rpms) if rpms else None, "vbat_samples": len(vbats)}


def monotonic(points: list[tuple[float, float]]) -> bool:
    return all(next_rpm >= rpm for (_power, rpm), (_next_power, next_rpm) in zip(points, points[1:]))


def slope_imbalance(slopes: list[float]) -> float | None:
    if len(slopes) != 4 or min(slopes) <= 0:
        return None
    mean = sum(slopes) / len(slopes)
    return max(abs(slope - mean) / mean for slope in slopes)


def vbat_range(rows: list[dict[str, float | int | None]]) -> dict[str, float | None]:
    values = [float(row["vbat"]) for row in rows if row["vbat"] is not None]
    return {"min": min(values) if values else None, "max": max(values) if values else None}


def _fmt(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def as_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def as_float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
