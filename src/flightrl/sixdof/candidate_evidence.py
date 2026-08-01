from __future__ import annotations

from typing import Any

from flightrl.evidence_scope import (
    DESKTOP_CPU_SCOPE,
    require_existing_file_identity,
    require_file_identity,
)
from flightrl.evidence_values import exact_true, failure_strings, finite_number
from flightrl.sixdof.controller import CONTROLLERS
from flightrl.sixdof.tasks import TASKS


REQUIRED_METRICS = (
    "mean_completed_fraction",
    "mean_position_error_m",
)
OPTIONAL_NONNEGATIVE_METRICS = (
    "mean_yaw_error_rad",
    "yaw_error_p95_rad",
    "teacher_action_l2_mean",
)


def validate_suite_record(record: object) -> None:
    if not isinstance(record, dict):
        raise ValueError("validation suite record must be an object")
    legacy = [field for field in ("edge_parity", "edge_latency") if field in record]
    if legacy:
        raise ValueError(f"legacy {', '.join(legacy)} evidence is non-authoritative")
    label = record.get("label")
    controller = record.get("controller")
    tasks = record.get("tasks")
    if not isinstance(label, str) or not label:
        raise ValueError("validation suite label is missing or invalid")
    if controller not in (*CONTROLLERS, "teacher"):
        raise ValueError(f"validation suite controller is invalid for {label}")
    if (
        not isinstance(tasks, list)
        or not tasks
        or not all(isinstance(task, str) for task in tasks)
        or len(tasks) != len(set(tasks))
        or any(task not in TASKS for task in tasks)
    ):
        raise ValueError(f"validation suite tasks are invalid for {label}")
    if controller != "teacher" and (not isinstance(record.get("checkpoint"), str) or not record["checkpoint"]):
        raise ValueError(f"validation suite checkpoint is invalid for {label}")
    validate_gate(record.get("gate"), f"validation suite gate for {label}")
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"validation suite metrics are invalid for {label}")
    validate_metrics(metrics, label)
    per_task = record.get("per_task_gate", {})
    if not isinstance(per_task, dict) or set(per_task) != set(tasks):
        raise ValueError(f"validation suite per-task gates are invalid for {label}")
    for task, gate in per_task.items():
        validate_gate(gate, f"validation suite {task} gate for {label}")
    if exact_true(record["gate"].get("passed")) and not all(
        exact_true(gate.get("passed")) for gate in per_task.values()
    ):
        raise ValueError(f"validation suite aggregate gate contradicts per-task gates for {label}")


def validate_gate(gate: object, label: str) -> None:
    if not isinstance(gate, dict) or type(gate.get("passed")) is not bool:
        raise ValueError(f"{label} is invalid")
    failures = failure_strings(gate.get("failures"))
    if failures is None or (exact_true(gate.get("passed")) and failures):
        raise ValueError(f"{label} is contradictory or invalid")


def validate_metrics(metrics: dict[str, Any], label: str) -> None:
    parsed = {key: finite_number(metrics.get(key)) for key in REQUIRED_METRICS}
    completion = parsed["mean_completed_fraction"]
    position_error = parsed["mean_position_error_m"]
    clearance_value = metrics.get("clearance_p01_m", metrics.get("min_clearance_m"))
    clearance = finite_number(clearance_value)
    survival = finite_number(metrics.get("mean_survival_fraction", completion))
    if (
        completion is None
        or not 0.0 <= completion <= 1.0
        or survival is None
        or not 0.0 <= survival <= 1.0
        or position_error is None
        or position_error < 0.0
        or clearance is None
        or clearance < 0.0
    ):
        raise ValueError(f"validation suite metrics are nonfinite or out of range for {label}")
    for key in OPTIONAL_NONNEGATIVE_METRICS:
        if key in metrics:
            value = finite_number(metrics[key])
            if value is None or value < 0.0:
                raise ValueError(f"validation suite metric {key} is invalid for {label}")
    if "action_saturation_fraction" in metrics:
        saturation = finite_number(metrics["action_saturation_fraction"])
        if saturation is None or not 0.0 <= saturation <= 1.0:
            raise ValueError(f"validation suite action saturation is invalid for {label}")


def validate_readiness_candidate(record: object) -> None:
    if not isinstance(record, dict):
        raise ValueError("candidate record must be an object")
    if not isinstance(record.get("label"), str) or not record["label"]:
        raise ValueError("candidate label is missing or invalid")
    if not isinstance(record.get("checkpoint"), str) or not record["checkpoint"]:
        raise ValueError("candidate checkpoint is missing or invalid")
    tasks = record.get("tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or not all(isinstance(task, str) for task in tasks)
        or len(tasks) != len(set(tasks))
        or any(task not in TASKS for task in tasks)
    ):
        raise ValueError("candidate tasks are missing or invalid")
    if record.get("controller") not in CONTROLLERS:
        raise ValueError("candidate controller is missing, retired, or invalid")
    if type(record.get("passed")) is not bool or failure_strings(record.get("failures")) is None:
        raise ValueError("candidate gate evidence is invalid")
    completion = finite_number(record.get("mean_completed_fraction"))
    position = finite_number(record.get("mean_position_error_m"))
    clearance = finite_number(record.get("clearance_p01_m"))
    if completion is None or not 0.0 <= completion <= 1.0 or position is None or position < 0.0 or clearance is None or clearance < 0.0:
        raise ValueError("candidate metrics are missing, nonfinite, or out of range")
    validate_candidate_desktop_evidence(record)
    per_task = record.get("per_task_gate", {})
    if not isinstance(per_task, dict) or set(per_task) != set(tasks):
        raise ValueError("candidate per-task gates are invalid")
    for task, gate in per_task.items():
        validate_gate(gate, f"candidate {task} gate")
    if exact_true(record.get("passed")) and not all(
        exact_true(gate.get("passed")) for gate in per_task.values()
    ):
        raise ValueError("candidate aggregate gate contradicts per-task gates")
    for key in ("mean_yaw_error_rad", "yaw_error_p95_rad"):
        if key in record and record[key] is not None:
            value = finite_number(record[key])
            if value is None or value < 0.0:
                raise ValueError(f"candidate metric {key} is invalid")


def validate_candidate_desktop_evidence(record: dict) -> None:
    for name in ("desktop_parity", "desktop_latency"):
        evidence = record.get(name)
        if not isinstance(evidence, dict) or evidence.get("evidence_scope") != DESKTOP_CPU_SCOPE or evidence.get("deployment_authority") is not False:
            raise ValueError(f"candidate {name} scope is invalid")
    parity = record["desktop_parity"]
    if exact_true(parity.get("passed")) and (
        not exact_true(parity.get("present"))
        or (error := finite_number(parity.get("max_abs_error"))) is None
        or error < 0.0
    ):
        raise ValueError("candidate desktop_parity pass evidence is invalid")


def validate_desktop_identities(record: dict, checkpoint: str) -> None:
    parity = record["desktop_parity"]
    if exact_true(parity.get("passed")):
        require_file_identity(parity.get("checkpoint"), checkpoint, label="desktop parity checkpoint")
        parity_model = require_existing_file_identity(parity.get("model"), label="desktop parity model")
    else:
        parity_model = None
    latency = record["desktop_latency"]
    if exact_true(latency.get("present")):
        require_file_identity(latency.get("checkpoint"), checkpoint, label="desktop latency checkpoint")
        latency_model = latency.get("model")
        if latency_model is not None:
            require_existing_file_identity(latency_model, label="desktop latency model")
            if parity_model is not None and latency_model != parity_model:
                raise ValueError("desktop parity and latency reference different models")


def compact_parity(report: dict | None, max_error: object) -> dict:
    threshold = finite_number(max_error)
    if threshold is None or threshold < 0.0:
        raise ValueError("max parity error must be a finite nonnegative number")
    if report is None:
        return missing_desktop_evidence(passed=False)
    parity = report.get("parity")
    error = finite_number(parity.get("max_abs_error")) if isinstance(parity, dict) else None
    if error is None or error < 0.0:
        raise ValueError("desktop parity max_abs_error must be a finite nonnegative number")
    observation = report.get("observation", {})
    if not isinstance(observation, dict):
        raise ValueError("desktop parity observation contract is invalid")
    return {
        "present": True,
        "passed": error <= threshold,
        "evidence_scope": DESKTOP_CPU_SCOPE,
        "deployment_authority": False,
        "max_abs_error": error,
        "model": report.get("model"),
        "checkpoint": report.get("checkpoint"),
        "observation_mode": observation.get("mode", "base"),
    }


def compact_latency(report: dict | None) -> dict:
    if report is None:
        return missing_desktop_evidence()
    key = "torchscript_result" if "torchscript_result" in report else "eager"
    result = report.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"desktop latency {key} is missing or invalid")
    per_sample = finite_number(result.get("per_sample_us"))
    samples_per_second = finite_number(result.get("samples_per_second"))
    if per_sample is None or per_sample <= 0.0 or samples_per_second is None or samples_per_second <= 0.0:
        raise ValueError("desktop latency metrics must be finite positive numbers")
    return {
        "present": True,
        "evidence_scope": DESKTOP_CPU_SCOPE,
        "deployment_authority": False,
        "per_sample_us": per_sample,
        "samples_per_second": samples_per_second,
        "checkpoint": report.get("checkpoint"),
        "model": report.get("torchscript") if key == "torchscript_result" else None,
    }


def missing_desktop_evidence(*, passed: bool | None = None) -> dict:
    result = {
        "present": False,
        "evidence_scope": DESKTOP_CPU_SCOPE,
        "deployment_authority": False,
    }
    if passed is not None:
        result["passed"] = passed
    return result
