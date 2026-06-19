from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightrl.sim2real.actuator import summarize_motor_bench
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
    hardware_blockers: list[str] | None = None,
    max_replay_state_rmse: float = 0.5,
    max_replay_range_rmse_mm: float = 300.0,
    min_motor_powers: int = 3,
) -> dict[str, Any]:
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


def summarize_calibration(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "ready": False, "failures": ["missing"]}
    summary = _read_json(path).get("summary", {})
    return {
        "present": True,
        "path": str(path),
        "ready": bool(summary.get("replay_calibration_ready", False)),
        "failures": summary.get("failures", []),
        "rows": summary.get("rows"),
        "duration_s": summary.get("duration_s"),
        "sample_rate_hz": summary.get("sample_rate_hz"),
        "floor_valid_ratio": summary.get("floor_valid_ratio"),
        "xy_span_m": summary.get("xy_span_m"),
        "yaw_span_deg": summary.get("yaw_span_deg"),
    }


def summarize_replay(path: Path | None, *, max_state_rmse: float, max_range_rmse_mm: float) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "passed": False, "failures": ["missing"]}
    aligned = _read_json(path).get("aligned", {})
    signals = aligned.get("signals", {})
    worst_state = _worst_rmse(signals, "stateEstimate.")
    worst_range = _worst_rmse(signals, "range.")
    failures = []
    if worst_state is None or worst_state > max_state_rmse:
        failures.append("state_rmse")
    if worst_range is None or worst_range > max_range_rmse_mm:
        failures.append("range_rmse")
    return {
        "present": True,
        "path": str(path),
        "passed": not failures,
        "failures": failures,
        "samples": aligned.get("samples"),
        "overlap_duration_s": aligned.get("overlap_duration_s"),
        "worst_state_rmse": worst_state,
        "worst_range_rmse_mm": worst_range,
    }


def summarize_stationary_noise(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "passed": False, "failures": ["missing"]}
    summary = _read_json(path).get("summary", {})
    return {
        "present": True,
        "path": str(path),
        "passed": bool(summary.get("stationary_noise_ready", False)),
        "failures": summary.get("failures", []),
        "duration_s": summary.get("duration_s"),
        "sample_rate_hz": summary.get("sample_rate_hz"),
        "max_position_span_m": summary.get("max_position_span_m"),
        "max_attitude_span_deg": summary.get("max_attitude_span_deg"),
    }


def summarize_hardware_latency(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "passed": False, "failures": ["missing"]}
    summary = _read_json(path).get("summary", {})
    return {
        "present": True,
        "path": str(path),
        "passed": bool(summary.get("latency_ready", False)),
        "failures": summary.get("failures", []),
        "accepted_pairs": summary.get("accepted_pairs"),
        "median_latency_s": summary.get("median_latency_s"),
    }


def summarize_deployment(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "passed": False, "failures": ["missing"]}
    data = _read_json(path)
    summary = data.get("summary", {})
    total = int(summary.get("total", 0) or 0)
    blocked = int(summary.get("blocked", 0) or 0)
    failures = []
    if total <= 0:
        failures.append("no_candidates")
    if blocked:
        failures.append("blocked_candidates")
    return {"present": True, "path": str(path), "passed": not failures, "failures": failures, "summary": summary}


def summarize_training_stack(deployment_readiness: Path | None) -> dict[str, Any]:
    if deployment_readiness is None or not deployment_readiness.exists():
        return {"present": False, "passed": False, "failures": ["readiness_missing"]}
    evidence = _read_json(deployment_readiness).get("global_evidence", {})
    throughput = evidence.get("training_throughput", {})
    puffer = evidence.get("puffer_export", {})
    failures = []
    if not throughput.get("present"):
        failures.append("training_throughput_missing")
    if not puffer.get("present") or not puffer.get("passed"):
        failures.append("puffer_export")
    return {"present": True, "passed": not failures, "failures": failures, "training_throughput": throughput, "puffer_export": puffer}


def collect_blockers(report: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not report["hardware_config"]["present"]:
        blockers.append("hardware_config_missing")
    elif not report["hardware_config"]["measured"]:
        blockers.append("measured_dynamics_missing")
    if report["hardware_config"].get("missing_parameters"):
        blockers.append("hardware_dynamics_incomplete")
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
    if not report["hardware_config"].get("sensor_model", {}).get("include_noisy_state"):
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
        ("deployment readiness", report["deployment_readiness"]["passed"], str(report["deployment_readiness"].get("summary", {})), "candidate gates"),
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _worst_rmse(signals: dict[str, Any], prefix: str) -> float | None:
    values = [float(metrics["rmse"]) for key, metrics in signals.items() if key.startswith(prefix) and "rmse" in metrics]
    return max(values) if values else None
