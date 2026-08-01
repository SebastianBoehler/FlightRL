from __future__ import annotations

from typing import Any

from flightrl.evidence_values import exact_nonnegative_int, exact_true, failure_strings, finite_number
from flightrl.sixdof.signal_evidence import RANGE_SIGNALS, REPLAY_STATE_SIGNALS, worst_complete_rmse


def compact_room(report: dict | None) -> dict:
    if report is None:
        return {"present": False, "mapping_ready": False, "failures": ["missing"]}
    if not isinstance(report, dict):
        raise ValueError("room report must be an object")
    summary = report.get("summary", {})
    estimate = report.get("room_estimate", {})
    failures = declared_failures(summary, "room")
    point_count = exact_nonnegative_int(summary.get("point_count")) if isinstance(summary, dict) else None
    duration = nonnegative(summary.get("duration_s")) if isinstance(summary, dict) else None
    dimensions = [nonnegative(estimate.get(key)) for key in ("width_m", "depth_m", "height_m")] if isinstance(estimate, dict) else []
    if point_count is None or point_count == 0 or duration is None or duration == 0.0 or len(dimensions) != 3 or any(value is None or value == 0.0 for value in dimensions):
        failures.append("room_metadata_invalid")
    return {
        "present": True,
        "mapping_ready": isinstance(summary, dict) and exact_true(summary.get("mapping_ready")) and not failures,
        "failures": failures,
        "point_count": point_count,
        "duration_s": duration,
        "width_m": dimensions[0] if len(dimensions) == 3 else None,
        "depth_m": dimensions[1] if len(dimensions) == 3 else None,
        "height_m": dimensions[2] if len(dimensions) == 3 else None,
        "warnings": valid_strings(estimate.get("warnings", [])) if isinstance(estimate, dict) else [],
    }


def compact_replay_comparison(report: dict | None, args: Any) -> dict:
    required = require_bool(args.require_replay_comparison, "require_replay_comparison")
    state_limit = require_nonnegative(args.max_replay_state_rmse, "max_replay_state_rmse")
    range_limit = require_nonnegative(args.max_replay_range_rmse_mm, "max_replay_range_rmse_mm")
    overlap_limit = require_nonnegative(args.min_replay_overlap_s, "min_replay_overlap_s")
    if report is None:
        return {"present": False, "required": required, "passed": not required}
    if not isinstance(report, dict):
        raise ValueError("replay comparison report must be an object")
    aligned = report.get("aligned", {})
    signals = aligned.get("signals", {}) if isinstance(aligned, dict) else {}
    samples = exact_nonnegative_int(aligned.get("samples")) if isinstance(aligned, dict) else None
    overlap = nonnegative(aligned.get("overlap_duration_s")) if isinstance(aligned, dict) else None
    worst_state = worst_complete_rmse(signals, REPLAY_STATE_SIGNALS)
    worst_range = worst_complete_rmse(signals, RANGE_SIGNALS)
    failures = []
    if samples is None or samples < 2 or overlap is None or overlap < overlap_limit:
        failures.append("overlap")
    if worst_state is None or worst_state > state_limit:
        failures.append("state_rmse")
    if worst_range is None or worst_range > range_limit:
        failures.append("range_rmse")
    return {
        "present": True,
        "required": required,
        "passed": not failures,
        "failures": failures,
        "samples": samples,
        "overlap_duration_s": overlap,
        "worst_state_rmse": worst_state,
        "worst_range_rmse_mm": worst_range,
    }


def declared_failures(summary: object, label: str) -> list[str]:
    if not isinstance(summary, dict):
        return [f"{label}_summary_invalid"]
    failures = failure_strings(summary.get("failures", []))
    return failures if failures is not None else [f"{label}_failures_invalid"]


def nonnegative(value: object) -> float | None:
    parsed = finite_number(value)
    return parsed if parsed is not None and parsed >= 0.0 else None


def require_nonnegative(value: object, name: str) -> float:
    parsed = nonnegative(value)
    if parsed is None:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return parsed


def require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def valid_strings(value: object) -> list[str]:
    return list(value) if isinstance(value, list) and all(isinstance(item, str) for item in value) else []
