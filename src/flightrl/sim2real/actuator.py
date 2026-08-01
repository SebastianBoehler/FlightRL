from __future__ import annotations

import csv
import json
from math import isfinite
from pathlib import Path
from typing import Any

from .actuator_fit import (
    fit_motor_curve,
    simulator_priors,
    slope_imbalance,
    validate_fit_thresholds,
)


MAX_PLAUSIBLE_RPM = 100_000.0


def summarize_motor_bench(path: Path | None, *, min_powers: int) -> dict[str, Any]:
    if type(min_powers) is not int or min_powers < 2:
        raise ValueError("min_powers must be an integer >= 2")
    if path is None or not path.exists():
        return {"present": False, "passed": False, "motors": {}, "failures": ["missing"]}
    rows, invalid_rows = parse_motor_rows(path)
    motors = group_motor_rows(rows)
    compact = {str(motor): motor_stats(values) for motor, values in sorted(motors.items())}
    failures = []
    if invalid_rows:
        failures.append("invalid_rows")
    if set(motors) != {1, 2, 3, 4}:
        failures.append("motor_coverage")
    if any(len({row["power"] for row in values}) < min_powers for values in motors.values()):
        failures.append("power_coverage")
    if any(not any(row["rpm"] > 0 for row in values) for values in motors.values()):
        failures.append("rpm_signal")
    if any(not any(row["vbat"] is not None for row in values) for values in motors.values()):
        failures.append("battery_signal")
    return {
        "present": True,
        "path": str(path),
        "passed": not failures,
        "motors": compact,
        "invalid_rows": invalid_rows,
        "failures": failures,
    }


def fit_motor_calibration(
    path: Path,
    *,
    min_powers: int = 3,
    min_r2: float = 0.9,
    max_gain_imbalance: float = 0.25,
    min_valid_rpm: float = 0.0,
    max_dropout_ratio: float = 0.0,
) -> dict[str, Any]:
    validate_fit_thresholds(
        min_powers=min_powers,
        min_r2=min_r2,
        max_gain_imbalance=max_gain_imbalance,
        min_valid_rpm=min_valid_rpm,
        max_dropout_ratio=max_dropout_ratio,
    )
    rows, invalid_rows = parse_motor_rows(path)
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
    if invalid_rows:
        failures.append("invalid_rows")
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
            "invalid_rows": invalid_rows,
            "warnings": ["rpm_dropouts_filtered"] if dropped_samples else [],
            "vbat": vbat_range(rows),
        },
        "motors": records,
        "simulator_priors": simulator_priors(records),
        "safety": "RPM calibration only; thrust/torque still require a thrust stand or replay fitting.",
    }


def load_motor_rows(path: Path) -> list[dict[str, float | int | None]]:
    rows, _invalid_rows = parse_motor_rows(path)
    return rows


def parse_motor_rows(
    path: Path,
) -> tuple[list[dict[str, float | int | None]], int]:
    rows = []
    invalid_rows = 0
    for row in csv.DictReader(path.open()):
        motor = as_int(row.get("motor"))
        power = as_float(row.get("power"))
        rpm = as_float(row.get("rpm"))
        if (
            motor is None
            or motor not in {1, 2, 3, 4}
            or power is None
            or not power.is_integer()
            or not 1.0 <= power <= 65_535.0
            or rpm is None
            or not 0.0 <= rpm <= MAX_PLAUSIBLE_RPM
        ):
            invalid_rows += 1
            continue
        vbat = as_float(row.get("vbat"))
        raw_vbat = row.get("vbat")
        if raw_vbat not in (None, "") and (
            vbat is None or not 0.0 < vbat < 10.0
        ):
            invalid_rows += 1
            continue
        rows.append(
            {
                "motor": motor,
                "power": power,
                "rpm": rpm,
                "vbat": vbat,
            }
        )
    return rows, invalid_rows


def group_motor_rows(rows: list[dict[str, float | int | None]]) -> dict[int, list[dict[str, float | int | None]]]:
    grouped: dict[int, list[dict[str, float | int | None]]] = {}
    for row in rows:
        grouped.setdefault(int(row["motor"]), []).append(row)
    return grouped


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
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None
