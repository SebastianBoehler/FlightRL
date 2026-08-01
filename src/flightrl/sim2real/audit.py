from __future__ import annotations

from pathlib import Path
from typing import Any

from flightrl.evidence_values import exact_true, failure_strings
from flightrl.sim2real.actuator import summarize_motor_bench
from flightrl.sim2real.audit_evidence import (
    finite_nonnegative,
    summarize_calibration,
    summarize_deployment,
    summarize_hardware_latency,
    summarize_replay,
    summarize_sensor_profile,
    summarize_stationary_noise,
    summarize_training_stack,
    valid_hardware_parameters,
)
from flightrl.sim2real.hardware_config import summarize_hardware_model


def build_audit(
    *,
    hardware_config: Path | None,
    calibration_quality: Path | None = None,
    deployment_readiness: Path | None = None,
    replay_comparison: Path | None = None,
    motor_bench: Path | None = None,
    stationary_noise: Path | None = None,
    hardware_latency: Path | None = None,
    sensor_profile: Path | None = None,
    hardware_blockers: list[str] | None = None,
    max_replay_state_rmse: float = 0.5,
    max_replay_range_rmse_mm: float = 300.0,
    min_motor_powers: int = 3,
) -> dict[str, Any]:
    validate_thresholds(
        max_replay_state_rmse=max_replay_state_rmse,
        max_replay_range_rmse_mm=max_replay_range_rmse_mm,
        min_motor_powers=min_motor_powers,
    )
    if hardware_blockers is not None and failure_strings(hardware_blockers) is None:
        raise ValueError("hardware_blockers must contain nonempty strings")
    report = {
        "hardware_config": summarize_hardware_config(hardware_config),
        "motor_bench": summarize_motor_bench(motor_bench, min_powers=min_motor_powers),
        "calibration_quality": summarize_calibration(calibration_quality),
        "replay_comparison": summarize_replay(
            replay_comparison,
            max_state_rmse=max_replay_state_rmse,
            max_range_rmse_mm=max_replay_range_rmse_mm,
        ),
        "stationary_noise": summarize_stationary_noise(stationary_noise),
        "hardware_latency": summarize_hardware_latency(hardware_latency),
        "sensor_profile": summarize_sensor_profile(sensor_profile),
        "deployment_readiness": summarize_deployment(deployment_readiness),
        "training_stack": summarize_training_stack(deployment_readiness),
        "thresholds": {
            "max_replay_state_rmse": max_replay_state_rmse,
            "max_replay_range_rmse_mm": max_replay_range_rmse_mm,
            "min_motor_powers": min_motor_powers,
        },
        "hardware_blockers": hardware_blockers or [],
        "safety": "Audit evidence only; not approval for live autonomous flight.",
    }
    blockers = collect_blockers(report)
    report["blocking_items"] = blockers
    report["transfer_ready"] = not blockers
    return report


def summarize_hardware_config(path: Path | None) -> dict[str, Any]:
    return summarize_hardware_model(path)


def collect_blockers(report: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not report["hardware_config"]["present"]:
        blockers.append("hardware_config_missing")
    elif not report["hardware_config"]["measured"]:
        blockers.append("measured_dynamics_missing")
    if report["hardware_config"].get("missing_parameters"):
        blockers.append("hardware_dynamics_incomplete")
    if not valid_hardware_parameters(report["hardware_config"].get("parameters")):
        blockers.append("hardware_dynamics_invalid")
    if not report["motor_bench"]["passed"]:
        blockers.append("motor_bench_missing" if not report["motor_bench"]["present"] else "motor_bench_failed")
    if not report["calibration_quality"]["ready"]:
        blockers.append("calibration_quality_missing" if not report["calibration_quality"]["present"] else "calibration_flight_not_ready")
    if not report["replay_comparison"]["passed"]:
        blockers.append("replay_comparison_missing" if not report["replay_comparison"]["present"] else "replay_comparison_failed")
    if not report["deployment_readiness"]["passed"]:
        blockers.append("deployment_readiness_blocked")
    if not report["training_stack"]["passed"]:
        blockers.append("training_stack_incomplete")
    sensor_model = report["hardware_config"].get("sensor_model", {})
    hardware_sensor_model = isinstance(sensor_model, dict) and exact_true(sensor_model.get("include_noisy_state"))
    measured_sensor_profile = exact_true(report.get("sensor_profile", {}).get("passed"))
    if not hardware_sensor_model and not measured_sensor_profile:
        blockers.append("sensor_model_incomplete")
    if not report["stationary_noise"]["passed"]:
        blockers.append("sensor_noise_unmeasured" if not report["stationary_noise"]["present"] else "sensor_noise_failed")
    if not report["hardware_latency"]["passed"]:
        blockers.append("latency_unmeasured" if not report["hardware_latency"]["present"] else "latency_failed")
    blockers.extend(report.get("hardware_blockers", []))
    return sorted(dict.fromkeys(blockers))


def render_markdown(report: dict[str, Any]) -> str:
    hardware_passed = report["hardware_config"]["present"] and report["hardware_config"].get("measured")
    rows = [
        ("hardware config", hardware_passed, report["hardware_config"].get("source"), "measured" if report["hardware_config"].get("measured") else "assumed"),
        ("motor bench", report["motor_bench"]["passed"], ",".join(report["motor_bench"].get("failures", [])) or "pass", "per-motor rpm/power/vbat"),
        ("calibration log", report["calibration_quality"]["ready"], ",".join(report["calibration_quality"].get("failures", [])) or "pass", "replay-ready flight"),
        ("replay comparison", report["replay_comparison"]["passed"], ",".join(report["replay_comparison"].get("failures", [])) or "pass", "real-vs-sim fit"),
        ("stationary noise", report["stationary_noise"]["passed"], ",".join(report["stationary_noise"].get("failures", [])) or "pass", "sensor randomization"),
        ("hardware latency", report["hardware_latency"]["passed"], ",".join(report["hardware_latency"].get("failures", [])) or "pass", "command/sensor timing"),
        ("sensor profile", report["sensor_profile"]["passed"], ",".join(report["sensor_profile"].get("failures", [])) or "pass", "sim observation/noise profile"),
        ("edge deployment authority", report["deployment_readiness"]["passed"], str(report["deployment_readiness"].get("summary", {})), "on-device candidate gates"),
        ("training stack", report["training_stack"]["passed"], ",".join(report["training_stack"].get("failures", [])) or "pass", "Puffer/export/throughput"),
    ]
    lines = ["# Sim-To-Real Audit", "", f"Transfer ready: `{report['transfer_ready']}`", "", "| area | passed | detail | scope |", "| --- | ---: | --- | --- |"]
    lines.extend(f"| {name} | {passed} | {detail} | {scope} |" for name, passed, detail, scope in rows)
    replay = report["replay_comparison"]
    lines.extend([
        "",
        "## Current Error",
        "",
        f"- Worst state RMSE: `{replay.get('worst_state_rmse')}` m",
        f"- Worst ranger RMSE: `{replay.get('worst_range_rmse_mm')}` mm",
        "",
        "## Blocking Items",
        "",
    ])
    lines.extend(f"- `{item}`" for item in report["blocking_items"])
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def validate_thresholds(
    *,
    max_replay_state_rmse: object,
    max_replay_range_rmse_mm: object,
    min_motor_powers: object,
) -> None:
    if (
        finite_nonnegative(max_replay_state_rmse) is None
        or finite_nonnegative(max_replay_range_rmse_mm) is None
    ):
        raise ValueError("replay RMSE thresholds must be finite nonnegative numbers")
    if type(min_motor_powers) is not int or min_motor_powers < 2:
        raise ValueError("min_motor_powers must be an integer >= 2")
