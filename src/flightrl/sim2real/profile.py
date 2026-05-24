from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    data["present"] = True
    data["path"] = str(path)
    return data


def collect_failures(hardware: dict[str, Any], motor: dict[str, Any], noise: dict[str, Any], latency: dict[str, Any]) -> list[str]:
    failures = []
    if not hardware["present"]:
        failures.append("hardware_config_missing")
    elif not hardware["measured"]:
        failures.append("measured_dynamics_missing")
    if any(value is None for value in hardware.get("parameters", {}).values()):
        failures.append("hardware_dynamics_incomplete")
    if not motor["present"]:
        failures.append("motor_calibration_missing")
    elif not motor.get("summary", {}).get("passed"):
        failures.append("motor_calibration_failed")
    if not noise["present"]:
        failures.append("stationary_noise_missing")
    elif not noise.get("summary", {}).get("stationary_noise_ready"):
        failures.append("stationary_noise_failed")
    if not latency["present"]:
        failures.append("hardware_latency_missing")
    elif not latency.get("summary", {}).get("latency_ready"):
        failures.append("hardware_latency_failed")
    return sorted(dict.fromkeys(failures))


def compact_motor(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    return {
        "present": report.get("present", False),
        "path": report.get("path"),
        "passed": bool(summary.get("passed", False)),
        "failures": summary.get("failures", []),
        "simulator_priors": report.get("simulator_priors", {}),
    }


def compact_noise(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    return {
        "present": report.get("present", False),
        "path": report.get("path"),
        "passed": bool(summary.get("stationary_noise_ready", False)),
        "failures": summary.get("failures", []),
        "duration_s": summary.get("duration_s"),
    }


def compact_latency(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    return {
        "present": report.get("present", False),
        "path": report.get("path"),
        "passed": bool(summary.get("latency_ready", False)),
        "failures": summary.get("failures", []),
        "median_latency_s": summary.get("median_latency_s"),
    }


def simulator_overlay(hardware: dict[str, Any], motor: dict[str, Any], noise: dict[str, Any], latency: dict[str, Any]) -> dict[str, Any]:
    return {
        "drone": hardware["parameters"],
        "actuator": motor["simulator_priors"],
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
    gain_imbalance = motor.get("summary", {}).get("gain_imbalance") or 0.0
    latency_s = latency.get("summary", {}).get("median_latency_s") or 0.0
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
    values = [signals[column]["std"] for column in columns if column in signals and signals[column].get("std") is not None]
    return float(max(values, default=0.0))


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
