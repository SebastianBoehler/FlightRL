from __future__ import annotations

import math

from flightrl.evidence_values import exact_true, failure_strings, finite_number
from flightrl.sixdof.controller import CONTROLLERS
from flightrl.sixdof.tasks import TASKS


def compact_profile_matrix(report: dict | None) -> dict:
    if report is None:
        return {"present": False, "valid": False, "profiles": [], "by_checkpoint": {}}
    if not isinstance(report, dict):
        raise ValueError("profile matrix report must be an object")
    profiles = report.get("profiles")
    raw_records = report.get("records")
    if (
        not isinstance(profiles, list)
        or not profiles
        or not all(isinstance(item, str) and item for item in profiles)
        or len(profiles) != len(set(profiles))
    ):
        raise ValueError("profile matrix profiles are missing or invalid")
    if not isinstance(raw_records, list):
        raise ValueError("profile matrix records are missing or invalid")
    records = [compact_profile_record(record, profiles) for record in raw_records]
    checkpoints = [record["checkpoint"] for record in records]
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError("profile matrix contains duplicate checkpoint records")
    return {
        "present": True,
        "valid": True,
        "profiles": profiles,
        "by_checkpoint": {record["checkpoint"]: record for record in records},
    }


def compact_profile_record(record: object, required_profiles: list[str]) -> dict:
    if not isinstance(record, dict):
        raise ValueError("profile matrix record must be an object")
    label = record.get("label")
    checkpoint = record.get("checkpoint")
    controller = record.get("controller")
    tasks = record.get("tasks")
    missing = record.get("missing_profiles")
    failures_by_profile = record.get("failures_by_profile")
    raw_profiles = record.get("profiles")
    if not isinstance(label, str) or not label or not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("profile matrix record identity is invalid")
    if controller not in CONTROLLERS:
        raise ValueError(f"profile matrix controller is retired or invalid for {label}")
    if (
        not isinstance(tasks, list)
        or not tasks
        or not all(isinstance(task, str) for task in tasks)
        or len(tasks) != len(set(tasks))
        or any(task not in TASKS for task in tasks)
    ):
        raise ValueError("profile matrix record tasks are invalid")
    if (
        not isinstance(missing, list)
        or not all(isinstance(item, str) for item in missing)
        or len(missing) != len(set(missing))
        or not all(item in required_profiles for item in missing)
    ):
        raise ValueError("profile matrix missing_profiles is invalid")
    if not isinstance(raw_profiles, dict) or not raw_profiles or set(raw_profiles) != set(required_profiles) - set(missing):
        raise ValueError("profile matrix profile evidence is incomplete")
    profiles = {name: compact_profile_evidence(value, name) for name, value in raw_profiles.items()}
    derived_failures = {
        name: value["failures"] for name, value in profiles.items() if not value["passed"]
    }
    if failures_by_profile != derived_failures:
        raise ValueError("profile matrix failures_by_profile contradicts profile evidence")
    passed = not missing and all(value["passed"] for value in profiles.values())
    if type(record.get("passed_all_profiles")) is not bool or exact_true(record.get("passed_all_profiles")) != passed:
        raise ValueError("profile matrix aggregate gate contradicts profile evidence")
    aggregates = aggregate_metrics(profiles.values())
    if not aggregate_matches(record, aggregates):
        raise ValueError("profile matrix aggregate metrics contradict profile evidence")
    return {
        "present": True,
        "label": label,
        "checkpoint": checkpoint,
        "controller": controller,
        "tasks": tasks,
        "profiles": profiles,
        "passed_all_profiles": passed,
        "missing_profiles": missing,
        "failures_by_profile": derived_failures,
        **aggregates,
    }


def compact_profile_evidence(value: object, name: str) -> dict:
    if not isinstance(value, dict) or type(value.get("passed")) is not bool:
        raise ValueError(f"profile evidence is invalid for {name}")
    failures = failure_strings(value.get("failures"))
    if failures is None or (exact_true(value.get("passed")) and failures) or (not value["passed"] and not failures):
        raise ValueError(f"profile gate is contradictory for {name}")
    completed = bounded_fraction(value.get("mean_completed_fraction"))
    survival = bounded_fraction(value.get("mean_survival_fraction"))
    position = nonnegative(value.get("mean_position_error_m"))
    clearance = nonnegative(value.get("clearance_p01_m"))
    yaw = nonnegative(value.get("mean_yaw_error_rad")) if value.get("mean_yaw_error_rad") is not None else None
    if any(item is None for item in (completed, survival, position, clearance)):
        raise ValueError(f"profile metrics are invalid for {name}")
    return {
        "passed": value["passed"],
        "failures": failures,
        "mean_completed_fraction": completed,
        "mean_survival_fraction": survival,
        "mean_position_error_m": position,
        "mean_yaw_error_rad": yaw,
        "clearance_p01_m": clearance,
    }


def aggregate_metrics(profiles) -> dict:
    values = list(profiles)
    return {
        "worst_survival_fraction": min(value["mean_survival_fraction"] for value in values),
        "worst_completed_fraction": min(value["mean_completed_fraction"] for value in values),
        "worst_position_error_m": max(value["mean_position_error_m"] for value in values),
        "worst_clearance_p01_m": min(value["clearance_p01_m"] for value in values),
        "worst_yaw_error_rad": max(
            (value["mean_yaw_error_rad"] for value in values if value["mean_yaw_error_rad"] is not None),
            default=None,
        ),
    }


def aggregate_matches(record: dict, aggregates: dict) -> bool:
    return all(same_optional_number(record.get(key), value) for key, value in aggregates.items())


def same_optional_number(value: object, expected: float | None) -> bool:
    if expected is None:
        return value is None
    parsed = finite_number(value)
    return parsed is not None and math.isclose(parsed, expected, rel_tol=1e-12, abs_tol=1e-12)


def profile_record(record: dict, profile_matrix: dict) -> dict:
    if not exact_true(profile_matrix.get("present")) or not exact_true(profile_matrix.get("valid")):
        return {"present": False}
    profile = profile_matrix.get("by_checkpoint", {}).get(record.get("checkpoint"))
    if (
        not isinstance(profile, dict)
        or profile.get("tasks") != record.get("tasks")
        or profile.get("controller") != record.get("controller")
    ):
        return {"present": False}
    return profile


def nonnegative(value: object) -> float | None:
    parsed = finite_number(value)
    return parsed if parsed is not None and parsed >= 0.0 else None


def bounded_fraction(value: object) -> float | None:
    parsed = finite_number(value)
    return parsed if parsed is not None and 0.0 <= parsed <= 1.0 else None
