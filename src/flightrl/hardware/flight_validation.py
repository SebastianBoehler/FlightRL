from __future__ import annotations

import csv
import json
from math import cos, hypot, isfinite, radians, sin
from pathlib import Path

from .flight_telemetry import (
    FLIGHT_TELEMETRY_VARIABLES,
    RANGER_FLIGHT_TELEMETRY_VARIABLES,
)
from .range_flight_validation import validate_range_flight


SCHEMA = "flightrl.instrumented_patrol_validation.v3"
PHASES = ("takeoff", "forward_1", "turn_left", "forward_2", "land", "complete")
OUT_AND_BACK_PHASES = ("takeoff", "forward", "backward", "land", "complete")
HEADER = ("host_time_s", "crazyflie_time_ms", *FLIGHT_TELEMETRY_VARIABLES)
RANGER_HEADER = (*HEADER, *RANGER_FLIGHT_TELEMETRY_VARIABLES)


def validate_instrumented_patrol(run_dir: str | Path) -> dict[str, object]:
    root = Path(run_dir)
    rows = _read_rows(root / "telemetry.csv")
    events = _read_events(root / "events.jsonl")
    phase_times = {event["phase"]: event["host_time_s"] for event in events}
    complete_phases = (
        tuple(event["phase"] for event in events) == PHASES
        and all(phase_times[PHASES[index]] < phase_times[PHASES[index + 1]] for index in range(len(PHASES) - 1))
    )

    forward_1 = _motion_segment(rows, phase_times, "forward_1", "turn_left")
    turn_left = _turn_segment(rows, phase_times)
    forward_2 = _motion_segment(rows, phase_times, "forward_2", "land")
    maximum_gap_s = max(
        (current["host_time_s"] - previous["host_time_s"] for previous, current in zip(rows, rows[1:])),
        default=float("inf"),
    )
    maximum_device_gap_ms = max(
        (
            current["crazyflie_time_ms"] - previous["crazyflie_time_ms"]
            for previous, current in zip(rows, rows[1:])
        ),
        default=float("inf"),
    )
    minimum_battery_v = min((row["pm.vbat"] for row in rows), default=float("-inf"))
    power_states = {row["pm.state"] for row in rows}
    final_altitude_m = rows[-1]["stateEstimate.z"] if rows else float("inf")

    checks = {
        "complete_phases": complete_phases,
        "forward_1": _forward_passed(forward_1),
        "forward_2": _forward_passed(forward_2),
        "landed": abs(final_altitude_m) <= 0.10,
        "telemetry_cadence": maximum_gap_s <= 0.25 and maximum_device_gap_ms <= 50.0,
        "telemetry_rows": len(rows) >= 350,
        "turn_left": 12.0 <= turn_left["yaw_delta_deg"] <= 28.0,
        "power_state": bool(rows)
        and all(state.is_integer() and int(state) in {0, 1, 2} for state in power_states),
    }
    range_report = None
    if rows and all(variable in rows[0] for variable in RANGER_FLIGHT_TELEMETRY_VARIABLES):
        range_report = validate_range_flight(rows)
        checks["range_calibration"] = range_report["range_calibration_passed"] is True
    passed = all(checks.values())
    metrics = {
        "rows": len(rows),
        "maximum_telemetry_gap_s": maximum_gap_s,
        "maximum_device_gap_ms": maximum_device_gap_ms,
        "minimum_battery_v": minimum_battery_v,
        "observed_power_states": sorted(power_states),
        "final_altitude_m": final_altitude_m,
        "forward_1": forward_1,
        "turn_left": turn_left,
        "forward_2": forward_2,
    }
    if range_report is not None:
        metrics["mapping"] = range_report["mapping"]
        metrics["flow"] = range_report["flow"]
    return {
        "schema": SCHEMA,
        "checks": checks,
        "failed_checks": [name for name, passed_check in checks.items() if not passed_check],
        "metrics": metrics,
        "instrumented_patrol_passed": passed,
        "range_calibration_passed": (
            range_report is not None
            and range_report["range_calibration_passed"] is True
        ),
        "range_calibration_failed_checks": (
            list(range_report["failed_checks"])
            if range_report is not None
            else ["missing_multiranger_telemetry"]
        ),
        "longer_scripted_stage_eligible": passed,
        "learned_policy_evaluated": False,
        "flight_authority": False,
    }


def validate_out_and_back(run_dir: str | Path) -> dict[str, object]:
    root = Path(run_dir)
    rows = _read_rows(root / "telemetry.csv")
    events = _read_events(root / "events.jsonl", phases=OUT_AND_BACK_PHASES)
    phase_times = {event["phase"]: event["host_time_s"] for event in events}
    complete = tuple(event["phase"] for event in events) == OUT_AND_BACK_PHASES
    forward_raw = _motion_segment(rows, phase_times, "forward", "backward")
    backward_raw = _motion_segment(rows, phase_times, "backward", "land")
    forward = _leg_metrics(forward_raw, direction=1.0)
    backward = _leg_metrics(backward_raw, direction=-1.0)
    forward_rows = _segment(rows, phase_times, "forward", "backward")
    backward_rows = _segment(rows, phase_times, "backward", "land")
    return_error_m = hypot(
        backward_rows[-1]["stateEstimate.x"] - forward_rows[0]["stateEstimate.x"],
        backward_rows[-1]["stateEstimate.y"] - forward_rows[0]["stateEstimate.y"],
    )
    host_gaps = [b["host_time_s"] - a["host_time_s"] for a, b in zip(rows, rows[1:])]
    device_gaps = [b["crazyflie_time_ms"] - a["crazyflie_time_ms"] for a, b in zip(rows, rows[1:])]
    distances = (forward["distance_m"], backward["distance_m"])
    repeatability = min(distances) / max(distances) if max(distances) > 0.0 else 0.0
    power_states = {row["pm.state"] for row in rows}
    checks = {
        "complete_phases": complete,
        "forward": _out_and_back_leg_passed(forward),
        "backward": _out_and_back_leg_passed(backward),
        "repeatability": repeatability >= 0.60,
        "returned": return_error_m <= 0.12,
        "landed": bool(rows) and abs(rows[-1]["stateEstimate.z"]) <= 0.10,
        "telemetry_cadence": bool(host_gaps)
        and max(host_gaps) <= 0.080
        and max(device_gaps) <= 50.0,
        "telemetry_rows": len(rows) >= 400,
        "power_state": bool(rows)
        and all(state.is_integer() and int(state) in {0, 1, 2} for state in power_states),
    }
    passed = all(checks.values())
    return {
        "schema": "flightrl.out_and_back_validation.v2",
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "metrics": {
            "rows": len(rows),
            "maximum_host_gap_s": max(host_gaps, default=float("inf")),
            "maximum_device_gap_ms": max(device_gaps, default=float("inf")),
            "minimum_battery_v": min((row["pm.vbat"] for row in rows), default=float("-inf")),
            "observed_power_states": sorted(power_states),
            "forward": forward,
            "backward": backward,
            "repeatability_ratio": repeatability,
            "return_error_m": return_error_m,
            "maximum_abs_roll_deg": max((abs(row["stateEstimate.roll"]) for row in rows), default=float("inf")),
            "maximum_abs_pitch_deg": max((abs(row["stateEstimate.pitch"]) for row in rows), default=float("inf")),
        },
        "out_and_back_passed": passed,
        "learned_policy_evaluated": False,
        "flight_authority": False,
    }
def _read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames not in {HEADER, RANGER_HEADER}:
            raise ValueError("instrumented patrol telemetry header is not exact")
        rows = []
        previous_host_time = float("-inf")
        previous_device_time = -1
        for raw in reader:
            row = {name: float(raw[name]) for name in fieldnames}
            if not all(isfinite(value) for value in row.values()):
                raise ValueError("instrumented patrol telemetry must be finite")
            host_time = row["host_time_s"]
            device_time = row["crazyflie_time_ms"]
            if host_time <= previous_host_time or device_time <= previous_device_time:
                raise ValueError("instrumented patrol telemetry must be ordered")
            previous_host_time = host_time
            previous_device_time = device_time
            rows.append(row)
    return rows


def _read_events(
    path: Path,
    *,
    phases: tuple[str, ...] = PHASES,
) -> list[dict[str, object]]:
    events = []
    for line in path.read_text().splitlines():
        raw = json.loads(line)
        phase = raw.get("phase")
        host_time_s = raw.get("host_time_s")
        if phase not in phases or not isinstance(host_time_s, (int, float)) or not isfinite(float(host_time_s)):
            raise ValueError("instrumented patrol event is invalid")
        events.append({"phase": phase, "host_time_s": float(host_time_s)})
    return events


def _segment(
    rows: list[dict[str, float]],
    phase_times: dict[str, float],
    start_phase: str,
    end_phase: str,
) -> list[dict[str, float]]:
    start = phase_times.get(start_phase, float("inf"))
    end = phase_times.get(end_phase, float("-inf"))
    selected = [row for row in rows if start <= row["host_time_s"] <= end]
    if len(selected) < 2:
        raise ValueError(f"instrumented patrol phase {start_phase} has too few rows")
    return selected


def _motion_segment(
    rows: list[dict[str, float]],
    phase_times: dict[str, float],
    start_phase: str,
    end_phase: str,
) -> dict[str, float]:
    segment = _segment(rows, phase_times, start_phase, end_phase)
    first, last = segment[0], segment[-1]
    dx = last["stateEstimate.x"] - first["stateEstimate.x"]
    dy = last["stateEstimate.y"] - first["stateEstimate.y"]
    yaw = radians(first["stateEstimate.yaw"])
    return {
        "forward_displacement_m": dx * cos(yaw) + dy * sin(yaw),
        "lateral_displacement_m": -dx * sin(yaw) + dy * cos(yaw),
        "minimum_altitude_m": min(row["stateEstimate.z"] for row in segment),
        "maximum_altitude_m": max(row["stateEstimate.z"] for row in segment),
    }


def _turn_segment(rows: list[dict[str, float]], phase_times: dict[str, float]) -> dict[str, float]:
    segment = _segment(rows, phase_times, "turn_left", "forward_2")
    delta = segment[-1]["stateEstimate.yaw"] - segment[0]["stateEstimate.yaw"]
    delta = (delta + 180.0) % 360.0 - 180.0
    return {"yaw_delta_deg": delta}


def _forward_passed(metrics: dict[str, float]) -> bool:
    return (
        0.18 <= metrics["forward_displacement_m"] <= 0.45
        and abs(metrics["lateral_displacement_m"]) <= 0.08
        and 0.25 <= metrics["minimum_altitude_m"]
        and metrics["maximum_altitude_m"] <= 0.55
    )


def _leg_metrics(metrics: dict[str, float], *, direction: float) -> dict[str, float]:
    return {
        "distance_m": direction * metrics["forward_displacement_m"],
        "lateral_displacement_m": metrics["lateral_displacement_m"],
        "minimum_altitude_m": metrics["minimum_altitude_m"],
        "maximum_altitude_m": metrics["maximum_altitude_m"],
    }


def _out_and_back_leg_passed(metrics: dict[str, float]) -> bool:
    return (
        0.30 <= metrics["distance_m"] <= 0.70
        and abs(metrics["lateral_displacement_m"]) <= 0.08
        and 0.25 <= metrics["minimum_altitude_m"]
        and metrics["maximum_altitude_m"] <= 0.55
    )
