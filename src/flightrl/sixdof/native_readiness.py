from __future__ import annotations

import math

from flightrl.evidence_values import exact_nonnegative_int
from flightrl.sixdof.signal_evidence import (
    NATIVE_STATE_SIGNALS,
    RANGE_SIGNALS,
    finite_nonnegative,
    signal_rmse,
    worst_complete_rmse,
)


def compact_native_parity(
    report: dict | None,
    max_state_rmse: float,
    max_range_rmse: float,
) -> dict:
    state_limit = require_nonnegative(max_state_rmse, "max_native_state_rmse")
    range_limit = require_nonnegative(max_range_rmse, "max_native_range_rmse")
    if report is None:
        return {"present": False, "passed": False, "failures": ["missing"]}
    if not isinstance(report, dict):
        raise ValueError("native parity report must be an object")
    aligned = report.get("aligned", {})
    signals = aligned.get("signals", {}) if isinstance(aligned, dict) else {}
    samples = exact_nonnegative_int(aligned.get("samples")) if isinstance(aligned, dict) else None
    overlap = finite_nonnegative(aligned.get("overlap_duration_s")) if isinstance(aligned, dict) else None
    failures = []
    if samples is None or samples < 2 or overlap is None or overlap == 0.0:
        failures.append("alignment_metadata")
    worst_state = worst_complete_rmse(signals, NATIVE_STATE_SIGNALS, detailed=True)
    worst_range = worst_complete_rmse(signals, RANGE_SIGNALS, detailed=True)
    if worst_state is None or worst_state > state_limit:
        failures.append("state_rmse")
    if worst_range is None or worst_range > range_limit:
        failures.append("range_rmse")
    profile_summary = validate_profiles(report, signals)
    if profile_summary is None:
        mismatches = None
        failures.append("profile_evidence_invalid")
    else:
        mismatches = profile_summary["mismatches"]
        if samples != profile_summary["samples"] or overlap is None or not math.isclose(
            overlap,
            profile_summary["duration_s"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            failures.append("profile_aggregate_mismatch")
        if mismatches:
            failures.append("termination_mismatch")
    return {
        "present": True,
        "passed": not failures,
        "failures": failures,
        "samples": samples,
        "overlap_duration_s": overlap,
        "worst_state_rmse": worst_state,
        "worst_range_rmse": worst_range,
        "termination_mismatches": mismatches,
    }


def validate_profiles(report: dict, aggregate_signals: object) -> dict | None:
    names = report.get("reset_profiles")
    profiles = report.get("profiles")
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or len(names) != len(set(names))
        or not isinstance(profiles, list)
        or len(profiles) != len(names)
    ):
        return None
    by_name = {}
    total_samples = 0
    total_duration = 0.0
    mismatches = 0
    for profile in profiles:
        if not isinstance(profile, dict):
            return None
        name = profile.get("reset_profile")
        samples = exact_nonnegative_int(profile.get("samples"))
        duration = finite_nonnegative(profile.get("duration_s"))
        terminal = exact_nonnegative_int(profile.get("terminal_mismatches"))
        truncation = exact_nonnegative_int(profile.get("truncation_mismatches"))
        signals = profile.get("signals")
        if (
            name not in names
            or name in by_name
            or samples is None
            or samples < 2
            or duration is None
            or duration == 0.0
            or terminal is None
            or truncation is None
            or worst_complete_rmse(signals, NATIVE_STATE_SIGNALS, detailed=True) is None
            or worst_complete_rmse(signals, RANGE_SIGNALS, detailed=True) is None
        ):
            return None
        by_name[name] = profile
        total_samples += samples
        total_duration += duration
        mismatches += terminal + truncation
    if set(by_name) != set(names) or not valid_aggregate_signals(aggregate_signals, by_name):
        return None
    return {"samples": total_samples, "duration_s": total_duration, "mismatches": mismatches}


def valid_aggregate_signals(signals: object, profiles: dict[str, dict]) -> bool:
    if not isinstance(signals, dict):
        return False
    for name in (*NATIVE_STATE_SIGNALS, *RANGE_SIGNALS):
        aggregate = signals.get(name)
        if signal_rmse(aggregate, detailed=True) is None or not isinstance(aggregate, dict):
            return False
        worst_profile = aggregate.get("worst_profile")
        if worst_profile not in profiles:
            return False
        values = {
            profile_name: signal_rmse(profile["signals"].get(name), detailed=True)
            for profile_name, profile in profiles.items()
        }
        worst = max(value for value in values.values() if value is not None)
        if aggregate.get("rmse") != worst or values[worst_profile] != worst:
            return False
    return True


def require_nonnegative(value: object, name: str) -> float:
    parsed = finite_nonnegative(value)
    if parsed is None:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return parsed
