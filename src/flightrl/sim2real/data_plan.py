from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightrl.evidence_values import exact_true, failure_strings
from flightrl.sim2real.actuator import summarize_motor_bench


REQUIREMENTS = {
    "actuator_curve": {
        "blockers": ["motor_bench_missing", "motor_bench_failed", "m3_motor_issue"],
        "acceptance": "All 4 motors recorded at >=3 power levels with positive RPM and battery voltage.",
        "command": "python scripts/crazyflie_motor_bench.py --confirm-props-off --output artifacts/crazyflie_logs/motor_bench.csv\npython scripts/fit_motor_bench.py --input artifacts/crazyflie_logs/motor_bench.csv --output artifacts/replay/motor_bench_calibration.json",
        "safety": "Props off; do not run while the M3 motor issue is unresolved.",
    },
    "measured_dynamics": {
        "blockers": ["measured_dynamics_missing", "hardware_dynamics_incomplete"],
        "acceptance": "Hardware config is derived from measured mass, arm length, actuator RPM curve, and replay-fitted drag/lag.",
        "command": None,
        "safety": (
            "No reviewed producer currently creates a measured hardware dynamics "
            "config. Treat manufacturer, simulator-alignment, and composed defaults "
            "as priors rather than transfer evidence."
        ),
    },
    "calibration_flight": {
        "blockers": ["calibration_quality_missing", "calibration_flight_not_ready"],
        "acceptance": "Calibration quality passes floor/ranger coverage, command modes, monotonic time, and sample-rate checks.",
        "command": None,
        "safety": "Blocked until a telemetry-watchdogged, supervisor-gated capture runner is implemented and reviewed.",
    },
    "replay_fit": {
        "blockers": ["replay_comparison_missing", "replay_comparison_failed", "deployment_readiness_blocked"],
        "acceptance": "Replay comparison passes state and ranger RMSE thresholds used by readiness gates.",
        "command": "python scripts/build_calibration_replay_report.py --log artifacts/crazyflie_logs/calibration_flight.csv --output-prefix artifacts/replay/calibration_flight",
        "safety": "A good replay fit promotes evidence; it still does not by itself approve live autonomous policy deployment.",
    },
    "sensor_noise": {
        "blockers": ["sensor_noise_unmeasured", "sensor_noise_failed"],
        "acceptance": "Stationary 60s telemetry estimates IMU/state/ranger noise and updates simulator randomization ranges.",
        "command": "python scripts/crazyflie_log.py --duration-s 60 --output artifacts/crazyflie_logs/stationary_noise.csv\npython scripts/summarize_stationary_noise.py --input artifacts/crazyflie_logs/stationary_noise.csv --output artifacts/replay/stationary_noise_summary.json",
        "safety": "Drone stationary on a level surface; no motor spin required.",
    },
    "latency": {
        "blockers": ["latency_unmeasured", "latency_failed"],
        "acceptance": "Command-to-estimator and sensor logging delays are measured or bounded for replay alignment.",
        "command": None,
        "safety": "Blocked until a dedicated bounded pulse collector exists; do not infer latency from unrelated logs.",
    },
}


def build_data_plan(audit_path: Path, *, motor_bench: Path | None = None) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text())
    parsed_blockers = failure_strings(audit.get("blocking_items", []))
    blockers = set(parsed_blockers or [])
    if parsed_blockers is None:
        blockers.add("audit_blockers_invalid")
    requirements = [requirement_record(name, spec, blockers) for name, spec in REQUIREMENTS.items()]
    report = {
        "audit": str(audit_path),
        "transfer_ready": exact_true(audit.get("transfer_ready")) and not blockers,
        "audit_blockers": sorted(blockers),
        "requirements": requirements,
        "partial_evidence": partial_evidence(motor_bench),
        "next_actions": [record for record in requirements if record["status"] != "satisfied"],
        "safety": "Do not run live hardware while the motor fault is unresolved or unattended.",
    }
    report["ready_to_collect_after_repair"] = ready_to_collect_after_repair(report)
    return report


def requirement_record(name: str, spec: dict[str, Any], blockers: set[str]) -> dict[str, Any]:
    matched = [blocker for blocker in spec["blockers"] if blocker in blockers]
    status = "blocked" if matched else "satisfied"
    if name in {"actuator_curve", "calibration_flight", "replay_fit"} and "m3_motor_issue" in blockers:
        status = "hardware_blocked"
    return {
        "name": name,
        "status": status,
        "matched_blockers": matched,
        "acceptance": spec["acceptance"],
        "command": spec["command"],
        "safety": spec["safety"],
    }


def partial_evidence(motor_bench: Path | None) -> dict[str, Any]:
    motor = summarize_motor_bench(motor_bench, min_powers=3) if motor_bench else {"present": False}
    evidence = {"motor_bench": motor}
    if motor.get("present") and not motor.get("passed"):
        evidence["motor_bench_note"] = "Existing motor bench evidence is partial or invalid for actuator calibration."
    return evidence


def ready_to_collect_after_repair(report: dict[str, Any]) -> list[str]:
    blocked_by_fault = {"actuator_curve", "calibration_flight", "replay_fit"}
    return [
        record["name"]
        for record in report["next_actions"]
        if record["name"] not in blocked_by_fault and record["status"] != "satisfied"
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sim-To-Real Data Plan",
        "",
        f"- Audit: `{report['audit']}`",
        f"- Transfer ready: `{report['transfer_ready']}`",
        f"- Audit blockers: `{', '.join(report['audit_blockers']) or 'none'}`",
        "",
        "| requirement | status | blockers | acceptance |",
        "| --- | --- | --- | --- |",
    ]
    for record in report["requirements"]:
        lines.append(
            f"| {record['name']} | {record['status']} | {', '.join(record['matched_blockers']) or 'none'} | {record['acceptance']} |"
        )
    lines.extend(["", "## Commands", ""])
    for record in report["next_actions"]:
        lines.extend([f"### {record['name']}", ""])
        if record["command"] is None:
            lines.append("No reviewed collection command is currently available.")
        else:
            lines.extend(["```bash", record["command"], "```"])
        lines.extend(["", f"Safety: {record['safety']}", ""])
    motor = report["partial_evidence"].get("motor_bench", {})
    if motor.get("present"):
        lines.extend(["## Partial Evidence", "", f"- Motor bench passed: `{motor.get('passed')}`; failures: `{', '.join(motor.get('failures', [])) or 'none'}`."])
    lines.extend(["", report["safety"]])
    return "\n".join(lines)
