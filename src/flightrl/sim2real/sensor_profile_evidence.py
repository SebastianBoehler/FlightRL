from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightrl.evidence_values import exact_true, failure_strings, finite_number
from flightrl.sim2real.live_profile import (
    build_live_sim_profile,
)


PROFILE_KNOBS = (
    "state_noise_std_m",
    "velocity_noise_std_m_s",
    "body_rate_noise_std_rad_s",
    "range_noise_std_m",
    "range_dropout_prob",
    "action_lag_s",
)


def summarize_sensor_profile(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "passed": False, "failures": ["missing"]}
    data = read_json(path)
    profile = data.get("sensor_profile")
    if not isinstance(profile, dict):
        return {"present": True, "path": str(path), "passed": False, "failures": ["invalid_profile"]}
    knobs = {key: finite_number(profile.get(key)) for key in PROFILE_KNOBS}
    failures = profile_summary_failures(data.get("summary"))
    if not any(key in profile for key in PROFILE_KNOBS):
        failures.append("empty_profile")
    if any(value is None or value < 0.0 for value in knobs.values()):
        failures.append("invalid_values")
    dropout = knobs["range_dropout_prob"]
    if dropout is None or dropout > 1.0:
        failures.append("invalid_dropout_probability")
    if not exact_true(profile.get("enabled")):
        failures.append("disabled")
    if not valid_profile_provenance(data):
        failures.append("profile_provenance_invalid")
    return {
        "present": True,
        "path": str(path),
        "passed": not failures,
        "failures": sorted(dict.fromkeys(failures)),
        "name": profile.get("name"),
        **knobs,
    }


def profile_summary_failures(summary: object) -> list[str]:
    if not isinstance(summary, dict):
        return ["profile_summary_invalid"]
    failures = failure_strings(summary.get("failures"))
    if failures is None:
        return ["profile_summary_invalid"]
    if not exact_true(summary.get("profile_ready")) or failures:
        return ["profile_not_ready", *failures]
    return []


def valid_profile_provenance(data: dict[str, Any]) -> bool:
    inputs = data.get("inputs")
    summary = data.get("summary")
    if not isinstance(inputs, dict) or not isinstance(summary, dict):
        return False
    flight_paths = evidence_paths(inputs.get("flight_logs"))
    stationary_paths = evidence_paths(inputs.get("stationary_logs"))
    if flight_paths is None or stationary_paths is None or not flight_paths:
        return False
    latency_value = inputs.get("latency_report")
    if latency_value is not None and (not isinstance(latency_value, str) or not Path(latency_value).is_file()):
        return False
    profile = data.get("sensor_profile")
    if not isinstance(profile, dict) or not isinstance(profile.get("name"), str):
        return False
    try:
        rebuilt = build_live_sim_profile(
            flight_logs=flight_paths,
            stationary_logs=stationary_paths,
            latency_report=Path(latency_value) if latency_value is not None else None,
            name=profile["name"],
        )
    except (OSError, TypeError, ValueError):
        return False
    return (
        exact_true(rebuilt["summary"].get("profile_ready"))
        and data.get("inputs") == rebuilt["inputs"]
        and summary == rebuilt["summary"]
        and profile == rebuilt["sensor_profile"]
    )


def evidence_paths(value: object) -> list[Path] | None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        return None
    paths = [Path(item) for item in value]
    return paths if all(path.is_file() for path in paths) else None


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}
