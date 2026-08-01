from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightrl.evidence_values import exact_nonnegative_int, exact_true, failure_strings, finite_number
from flightrl.sim2real.deployment_evidence import deployment_contract_failures
from flightrl.sim2real.noise import (
    DEFAULT_COLUMNS,
    MIN_SIGNAL_SAMPLES,
    MIN_SIGNAL_VALID_RATIO,
    MIN_STATIONARY_ROWS,
    MIN_STATIONARY_SAMPLE_RATE_HZ,
)
from flightrl.sim2real.sensor_profile_evidence import summarize_sensor_profile as _summarize_sensor_profile
from flightrl.sixdof.signal_evidence import RANGE_SIGNALS, REPLAY_STATE_SIGNALS, worst_complete_rmse


def summarize_calibration(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "ready": False, "failures": ["missing"]}
    raw_summary = read_json(path).get("summary", {})
    failures = summary_failures(raw_summary, "calibration_quality")
    summary = raw_summary if isinstance(raw_summary, dict) else {}
    rows = exact_nonnegative_int(summary.get("rows"))
    duration_s = finite_nonnegative(summary.get("duration_s"))
    sample_rate_hz = finite_nonnegative(summary.get("sample_rate_hz"))
    if rows is None or rows == 0 or duration_s is None or duration_s == 0.0 or sample_rate_hz is None or sample_rate_hz == 0.0:
        failures.append("calibration_quality_invalid_metrics")
    return {
        "present": True,
        "path": str(path),
        "ready": exact_true(summary.get("replay_calibration_ready")) and not failures,
        "failures": failures,
        "rows": rows,
        "duration_s": duration_s,
        "sample_rate_hz": sample_rate_hz,
        "floor_valid_ratio": summary.get("floor_valid_ratio"),
        "xy_span_m": summary.get("xy_span_m"),
        "yaw_span_deg": summary.get("yaw_span_deg"),
    }


def summarize_replay(path: Path | None, *, max_state_rmse: float, max_range_rmse_mm: float) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "passed": False, "failures": ["missing"]}
    aligned = read_json(path).get("aligned", {})
    if not isinstance(aligned, dict):
        aligned = {}
    signals = aligned.get("signals", {})
    worst_state = worst_complete_rmse(signals, REPLAY_STATE_SIGNALS)
    worst_range = worst_complete_rmse(signals, RANGE_SIGNALS)
    samples = exact_nonnegative_int(aligned.get("samples"))
    overlap_duration_s = finite_nonnegative(aligned.get("overlap_duration_s"))
    failures = []
    if samples is None or samples < 2 or overlap_duration_s is None or overlap_duration_s == 0.0:
        failures.append("replay_metadata")
    if worst_state is None or worst_state > max_state_rmse:
        failures.append("state_rmse")
    if worst_range is None or worst_range > max_range_rmse_mm:
        failures.append("range_rmse")
    return {
        "present": True,
        "path": str(path),
        "passed": not failures,
        "failures": failures,
        "samples": samples,
        "overlap_duration_s": overlap_duration_s,
        "worst_state_rmse": worst_state,
        "worst_range_rmse_mm": worst_range,
    }


def summarize_stationary_noise(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "passed": False, "failures": ["missing"]}
    data = read_json(path)
    raw_summary = data.get("summary", {})
    failures = summary_failures(raw_summary, "stationary_noise")
    summary = raw_summary if isinstance(raw_summary, dict) else {}
    duration_s = finite_nonnegative(summary.get("duration_s"))
    sample_rate_hz = finite_nonnegative(summary.get("sample_rate_hz"))
    rows = exact_nonnegative_int(summary.get("rows"))
    position_span = finite_nonnegative(summary.get("max_position_span_m"))
    attitude_span = finite_nonnegative(summary.get("max_attitude_span_deg"))
    if (
        duration_s is None
        or duration_s == 0.0
        or sample_rate_hz is None
        or sample_rate_hz < MIN_STATIONARY_SAMPLE_RATE_HZ
        or rows is None
        or rows < MIN_STATIONARY_ROWS
        or position_span is None
        or attitude_span is None
        or not valid_stationary_signals(data.get("signals"))
    ):
        failures.append("stationary_noise_invalid_metrics")
    return {
        "present": True,
        "path": str(path),
        "passed": exact_true(summary.get("stationary_noise_ready")) and not failures,
        "failures": failures,
        "duration_s": duration_s,
        "rows": rows,
        "sample_rate_hz": sample_rate_hz,
        "max_position_span_m": position_span,
        "max_attitude_span_deg": attitude_span,
    }


def valid_stationary_signals(signals: object) -> bool:
    if not isinstance(signals, dict):
        return False
    for column in DEFAULT_COLUMNS:
        signal = signals.get(column)
        if not isinstance(signal, dict):
            return False
        samples = exact_nonnegative_int(signal.get("samples"))
        valid_ratio = finite_nonnegative(signal.get("valid_ratio"))
        std = finite_nonnegative(signal.get("std"))
        if (
            samples is None
            or samples < MIN_SIGNAL_SAMPLES
            or valid_ratio is None
            or valid_ratio < MIN_SIGNAL_VALID_RATIO
            or valid_ratio > 1.0
            or std is None
        ):
            return False
    return True


def summarize_hardware_latency(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "passed": False, "failures": ["missing"]}
    raw_summary = read_json(path).get("summary", {})
    failures = summary_failures(raw_summary, "hardware_latency")
    summary = raw_summary if isinstance(raw_summary, dict) else {}
    accepted_pairs = exact_nonnegative_int(summary.get("accepted_pairs"))
    median_latency_s = finite_nonnegative(summary.get("median_latency_s"))
    if accepted_pairs is None or accepted_pairs == 0 or median_latency_s is None:
        failures.append("hardware_latency_invalid_metrics")
    return {
        "present": True,
        "path": str(path),
        "passed": exact_true(summary.get("latency_ready")) and not failures,
        "failures": failures,
        "accepted_pairs": accepted_pairs,
        "median_latency_s": median_latency_s,
    }


def summarize_sensor_profile(path: Path | None) -> dict[str, Any]:
    return _summarize_sensor_profile(path)


def summarize_deployment(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "passed": False, "failures": ["missing"]}
    data = read_json(path)
    authority_failures = deployment_contract_failures(data)
    raw_summary = data.get("summary", {})
    failures = [
        *authority_failures,
        *summary_failures(raw_summary, "deployment_readiness"),
    ]
    summary = raw_summary if isinstance(raw_summary, dict) else {}
    total = exact_nonnegative_int(summary.get("total"))
    ready = exact_nonnegative_int(summary.get("ready"))
    blocked = exact_nonnegative_int(summary.get("blocked"))
    derived = readiness_counts(data.get("records"))
    if total is None or ready is None or blocked is None or derived is None or (total, ready, blocked) != derived:
        failures.append("invalid_readiness_summary")
    elif total == 0:
        failures.append("no_candidates")
    elif blocked:
        failures.append("blocked_candidates")
    return {
        "present": True,
        "path": str(path),
        "passed": not failures,
        "failures": failures,
        "summary": summary,
        "evidence_scope": data.get("evidence_scope"),
        "deployment_authority": False,
    }


def summarize_training_stack(deployment_readiness: Path | None) -> dict[str, Any]:
    if deployment_readiness is None or not deployment_readiness.exists():
        return {"present": False, "passed": False, "failures": ["readiness_missing"]}
    evidence = read_json(deployment_readiness).get("global_evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    throughput = evidence.get("training_throughput", {})
    puffer = evidence.get("puffer_export", {})
    throughput = throughput if isinstance(throughput, dict) else {}
    puffer = puffer if isinstance(puffer, dict) else {}
    best = throughput.get("best_total_sps") or {}
    best = best if isinstance(best, dict) else {}
    total_sps = finite_nonnegative(best.get("total_sps"))
    failures = []
    if not exact_true(throughput.get("present")) or not exact_true(throughput.get("valid")) or total_sps is None or total_sps == 0.0:
        failures.append("training_throughput_missing")
    checks = puffer.get("checks")
    puffer_failures = failure_strings(puffer.get("failures", []))
    puffer_valid = (
        exact_true(puffer.get("present"))
        and exact_true(puffer.get("passed"))
        and isinstance(puffer.get("env_name"), str)
        and bool(puffer["env_name"])
        and isinstance(checks, list)
        and bool(checks)
        and puffer_failures == []
        and all(
            isinstance(check, dict)
            and exact_true(check.get("passed"))
            and failure_strings(check.get("failures", [])) == []
            for check in checks
        )
    )
    if not puffer_valid:
        failures.append("puffer_export")
    return {
        "present": True,
        "passed": not failures,
        "failures": failures,
        "training_throughput": throughput,
        "puffer_export": puffer,
    }


def valid_hardware_parameters(parameters: object) -> bool:
    if not isinstance(parameters, dict):
        return False
    nonnegative = {"drag", "angular_drag"}
    keys = (
        "mass", "inertia", "arm_length", "drag", "angular_drag",
        "hover_thrust", "thrust_gain", "max_total_thrust", "max_pitch_torque", "actuator_tau",
    )
    for key in keys:
        value = finite_number(parameters.get(key))
        if value is None or value < 0.0 or (key not in nonnegative and value == 0.0):
            return False
    return True


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}


def finite_nonnegative(value: object) -> float | None:
    parsed = finite_number(value)
    return parsed if parsed is not None and parsed >= 0.0 else None


def summary_failures(summary: object, label: str) -> list[str]:
    if not isinstance(summary, dict):
        return [f"{label}_invalid_summary"]
    failures = failure_strings(summary.get("failures", []))
    return failures if failures is not None else [f"{label}_invalid_failures"]


def readiness_counts(records: object) -> tuple[int, int, int] | None:
    if not isinstance(records, list):
        return None
    ready = 0
    for record in records:
        if (
            not isinstance(record, dict)
            or type(record.get("ready")) is not bool
            or not isinstance(record.get("task"), str)
            or not record["task"]
        ):
            return None
        failures = failure_strings(record.get("failures"))
        if failures is None or (record["ready"] and failures):
            return None
        ready += int(record["ready"])
    return len(records), ready, len(records) - ready
