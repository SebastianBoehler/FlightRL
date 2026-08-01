from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightrl.evidence_values import exact_nonnegative_int, exact_true, failure_strings, finite_number
from flightrl.sim2real.audit_evidence import valid_hardware_parameters, valid_stationary_signals
from flightrl.sim2real.noise import MIN_STATIONARY_ROWS, MIN_STATIONARY_SAMPLE_RATE_HZ
from flightrl.sim2real.hardware_config import summarize_hardware_model


STATE_COLUMNS = ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z"]
ATTITUDE_COLUMNS = ["stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw"]
IMU_COLUMNS = ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]
RANGE_COLUMNS = ["range.front", "range.back", "range.left", "range.right", "range.up", "range.zrange"]


def build_profile(
    *,
    hardware_config: Path,
    motor_calibration: Path | None,
    stationary_noise: Path | None,
    hardware_latency: Path | None,
) -> dict[str, Any]:
    hardware = summarize_hardware(hardware_config)
    motor = read_evidence(motor_calibration, "motor_calibration")
    noise = read_evidence(stationary_noise, "stationary_noise")
    latency = read_evidence(hardware_latency, "hardware_latency")
    failures = collect_failures(hardware, motor, noise, latency)
    report = {
        "hardware_config": hardware,
        "evidence": {
            "motor_calibration": compact_motor(motor),
            "stationary_noise": compact_noise(noise),
            "hardware_latency": compact_latency(latency),
        },
        "summary": {"profile_ready": not failures, "failures": failures},
        "safety": "Simulator profile only; live policy deployment still requires replay and readiness gates.",
    }
    if not failures:
        report["simulator_overlay"] = simulator_overlay(hardware, motor, noise, latency)
    return report


def summarize_hardware(path: Path) -> dict[str, Any]:
    return summarize_hardware_model(path)


def read_evidence(path: Path | None, key: str) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "kind": key}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        return {"present": True, "kind": key, "path": str(path), "invalid": True}
    data["present"] = True
    data["path"] = str(path)
    return data


def collect_failures(hardware: dict[str, Any], motor: dict[str, Any], noise: dict[str, Any], latency: dict[str, Any]) -> list[str]:
    failures = []
    if not hardware["present"]:
        failures.append("hardware_config_missing")
    elif not hardware["measured"]:
        failures.append("measured_dynamics_missing")
    parameters = hardware.get("parameters", {})
    if not isinstance(parameters, dict) or any(value is None for value in parameters.values()):
        failures.append("hardware_dynamics_incomplete")
    if not valid_hardware_parameters(parameters):
        failures.append("hardware_dynamics_invalid")
    if not motor["present"]:
        failures.append("motor_calibration_missing")
    elif not evidence_passed(motor, "passed") or not valid_motor_evidence(motor):
        failures.append("motor_calibration_failed")
    if not noise["present"]:
        failures.append("stationary_noise_missing")
    elif not evidence_passed(noise, "stationary_noise_ready") or not valid_noise_evidence(noise):
        failures.append("stationary_noise_failed")
    if not latency["present"]:
        failures.append("hardware_latency_missing")
    elif not evidence_passed(latency, "latency_ready") or not valid_latency_evidence(latency):
        failures.append("hardware_latency_failed")
    return sorted(dict.fromkeys(failures))


def compact_motor(report: dict[str, Any]) -> dict[str, Any]:
    raw_summary = report.get("summary", {})
    failures = compact_failures(raw_summary, "motor_calibration")
    if not valid_motor_evidence(report):
        failures.append("motor_calibration_invalid_metrics")
    return {
        "present": report.get("present", False),
        "path": report.get("path"),
        "passed": evidence_passed(report, "passed") and valid_motor_evidence(report),
        "failures": sorted(dict.fromkeys(failures)),
        "simulator_priors": report.get("simulator_priors", {}),
    }


def compact_noise(report: dict[str, Any]) -> dict[str, Any]:
    raw_summary = report.get("summary", {})
    summary = raw_summary if isinstance(raw_summary, dict) else {}
    failures = compact_failures(raw_summary, "stationary_noise")
    if not valid_noise_evidence(report):
        failures.append("stationary_noise_invalid_metrics")
    return {
        "present": report.get("present", False),
        "path": report.get("path"),
        "passed": evidence_passed(report, "stationary_noise_ready") and valid_noise_evidence(report),
        "failures": sorted(dict.fromkeys(failures)),
        "duration_s": summary.get("duration_s"),
    }


def compact_latency(report: dict[str, Any]) -> dict[str, Any]:
    raw_summary = report.get("summary", {})
    summary = raw_summary if isinstance(raw_summary, dict) else {}
    failures = compact_failures(raw_summary, "hardware_latency")
    if not valid_latency_evidence(report):
        failures.append("hardware_latency_invalid_metrics")
    return {
        "present": report.get("present", False),
        "path": report.get("path"),
        "passed": evidence_passed(report, "latency_ready") and valid_latency_evidence(report),
        "failures": sorted(dict.fromkeys(failures)),
        "median_latency_s": summary.get("median_latency_s"),
    }


def simulator_overlay(hardware: dict[str, Any], motor: dict[str, Any], noise: dict[str, Any], latency: dict[str, Any]) -> dict[str, Any]:
    return {
        "drone": hardware["parameters"],
        "actuator": compact_actuator_priors(motor),
        "sensors": {
            "state_noise_std": max_std(noise, STATE_COLUMNS),
            "attitude_noise_std_deg": max_std(noise, ATTITUDE_COLUMNS),
            "imu_noise_std": max_std(noise, IMU_COLUMNS),
            "range_noise_std_mm": max_std(noise, RANGE_COLUMNS),
            "command_latency_s": latency["summary"]["median_latency_s"],
        },
        "domain_randomization": recommend_randomization(motor, noise, latency),
    }


def recommend_randomization(motor: dict[str, Any], noise: dict[str, Any], latency: dict[str, Any]) -> dict[str, Any]:
    gain_imbalance = finite_number(motor.get("summary", {}).get("gain_imbalance"))
    latency_s = finite_number(latency.get("summary", {}).get("median_latency_s"))
    assert gain_imbalance is not None and latency_s is not None
    return {
        "enabled": True,
        "motor_gain_scale": max(0.02, float(gain_imbalance)),
        "sensor_noise_scale": 1.0,
        "latency_s": latency_s,
        "state_noise_std": max_std(noise, STATE_COLUMNS),
        "range_noise_std_mm": max_std(noise, RANGE_COLUMNS),
    }


def max_std(report: dict[str, Any], columns: list[str]) -> float:
    signals = report.get("signals", {})
    values = [finite_number(signals[column].get("std")) for column in columns if isinstance(signals.get(column), dict)]
    valid = [value for value in values if value is not None and value >= 0.0]
    return max(valid, default=0.0)


def evidence_passed(report: dict[str, Any], flag: str) -> bool:
    summary = report.get("summary", {})
    failures = failure_strings(summary.get("failures", [])) if isinstance(summary, dict) else None
    return isinstance(summary, dict) and exact_true(summary.get(flag)) and failures == []


def compact_failures(summary: object, label: str) -> list[str]:
    if not isinstance(summary, dict):
        return [f"{label}_invalid_summary"]
    failures = failure_strings(summary.get("failures", []))
    return failures if failures is not None else [f"{label}_invalid_failures"]


def valid_motor_evidence(report: dict[str, Any]) -> bool:
    summary = report.get("summary", {})
    priors = report.get("simulator_priors", {})
    gains = priors.get("relative_motor_gains", {}) if isinstance(priors, dict) else {}
    gain_imbalance = finite_number(summary.get("gain_imbalance")) if isinstance(summary, dict) else None
    slope = finite_number(priors.get("mean_slope_rpm_per_power")) if isinstance(priors, dict) else None
    gain_values = [finite_number(gains.get(str(motor))) for motor in range(1, 5)] if isinstance(gains, dict) else []
    return (
        isinstance(priors, dict)
        and exact_true(priors.get("present"))
        and gain_imbalance is not None
        and gain_imbalance >= 0.0
        and slope is not None
        and slope > 0.0
        and len(gain_values) == 4
        and all(value is not None and value > 0.0 for value in gain_values)
    )


def valid_noise_evidence(report: dict[str, Any]) -> bool:
    summary = report.get("summary", {})
    duration = finite_number(summary.get("duration_s")) if isinstance(summary, dict) else None
    sample_rate = finite_number(summary.get("sample_rate_hz")) if isinstance(summary, dict) else None
    rows = exact_nonnegative_int(summary.get("rows")) if isinstance(summary, dict) else None
    signals = report.get("signals", {})
    columns = STATE_COLUMNS + ATTITUDE_COLUMNS + IMU_COLUMNS + RANGE_COLUMNS
    stds = [finite_number(signals.get(column, {}).get("std")) for column in columns if isinstance(signals.get(column), dict)] if isinstance(signals, dict) else []
    return (
        duration is not None
        and duration > 0.0
        and sample_rate is not None
        and sample_rate >= MIN_STATIONARY_SAMPLE_RATE_HZ
        and rows is not None
        and rows >= MIN_STATIONARY_ROWS
        and len(stds) == len(columns)
        and all(value is not None and value >= 0.0 for value in stds)
        and valid_stationary_signals(signals)
    )


def valid_latency_evidence(report: dict[str, Any]) -> bool:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return False
    accepted = exact_nonnegative_int(summary.get("accepted_pairs"))
    latency = finite_number(summary.get("median_latency_s"))
    return accepted is not None and accepted > 0 and latency is not None and latency >= 0.0


def compact_actuator_priors(report: dict[str, Any]) -> dict[str, Any]:
    priors = report["simulator_priors"]
    return {
        "present": True,
        "mean_slope_rpm_per_power": finite_number(priors["mean_slope_rpm_per_power"]),
        "relative_motor_gains": {
            str(motor): finite_number(priors["relative_motor_gains"][str(motor)])
            for motor in range(1, 5)
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sim-To-Real Profile",
        "",
        f"- Profile ready: `{report['summary']['profile_ready']}`",
        f"- Failures: `{', '.join(report['summary']['failures']) or 'none'}`",
        f"- Hardware: `{report['hardware_config'].get('path')}`",
        "",
        "| evidence | present | passed | failures |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, evidence in report["evidence"].items():
        lines.append(f"| {name} | {evidence['present']} | {evidence['passed']} | {', '.join(evidence.get('failures', [])) or 'none'} |")
    if "simulator_overlay" in report:
        lines.extend(["", "## Simulator Overlay", "", "```json", json.dumps(report["simulator_overlay"], indent=2, sort_keys=True), "```"])
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
