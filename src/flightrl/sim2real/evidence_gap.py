from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightrl.sim2real.provenance import path_provenance


CATEGORY_BY_BLOCKER = {
    "m3_motor_issue": "hardware_repair",
    "hardware_config_missing": "dynamics_model",
    "measured_dynamics_missing": "dynamics_model",
    "hardware_dynamics_incomplete": "dynamics_model",
    "motor_bench_missing": "actuator_model",
    "motor_bench_failed": "actuator_model",
    "calibration_quality_missing": "calibration_flight",
    "calibration_flight_not_ready": "calibration_flight",
    "replay_comparison_missing": "replay_validation",
    "replay_comparison_failed": "replay_validation",
    "sensor_noise_unmeasured": "sensing_and_timing",
    "sensor_noise_failed": "sensing_and_timing",
    "latency_unmeasured": "sensing_and_timing",
    "latency_failed": "sensing_and_timing",
    "deployment_readiness_blocked": "policy_deployment",
    "training_stack_incomplete": "policy_deployment",
    "sensor_model_incomplete": "sensor_model",
}

ACTION_BY_CATEGORY = {
    "hardware_repair": "Resolve the motor/rotor issue before collecting transfer evidence or running learned policies.",
    "dynamics_model": "Replace manufacturer placeholder dynamics with measured mass, inertia, thrust, drag, and actuator constants.",
    "actuator_model": "Collect a clean per-motor bench over multiple power levels, including RPM, requested output, actual output, and battery voltage.",
    "sensing_and_timing": "Measure stationary sensor noise and command-to-observation latency for domain randomization and replay alignment.",
    "calibration_flight": "Record a supervised calibration flight with enough floor, XY, yaw, and ranger coverage.",
    "replay_validation": "Replay the calibration commands in sim and require state/ranger RMSE below the configured thresholds.",
    "policy_deployment": "Keep checkpoints sim-only until deployment readiness, export parity, latency, and replay gates all pass.",
    "sensor_model": "Model noisy state and ranger observations instead of training against clean privileged state only.",
    "uncategorized": "Inspect uncategorized blockers and either add evidence or extend the gate taxonomy.",
}


def build_evidence_gap_report(pipeline: Path) -> dict[str, Any]:
    data = json.loads(pipeline.read_text())
    blockers = list(data.get("blocking_items", []))
    categories = categorize(blockers)
    enough = (
        bool(data.get("transfer_approved", False))
        and int(data.get("hardware_approved_checkpoints", 0) or 0) > 0
        and not blockers
    )
    return {
        "pipeline": path_provenance(pipeline),
        "enough_for_one_step_transfer": enough,
        "decision": "ready_for_supervised_transfer_test" if enough else "blocked",
        "categories": categories,
        "action_items": action_items(categories),
        "transfer_approved": bool(data.get("transfer_approved", False)),
        "hardware_approved_checkpoints": int(data.get("hardware_approved_checkpoints", 0) or 0),
        "safety": "This report is offline evidence triage. It does not approve live autonomous flight.",
    }


def categorize(blockers: list[str]) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for blocker in blockers:
        category = CATEGORY_BY_BLOCKER.get(blocker, "uncategorized")
        categories.setdefault(category, []).append(blocker)
    return {key: sorted(values) for key, values in sorted(categories.items())}


def action_items(categories: dict[str, list[str]]) -> list[str]:
    return [ACTION_BY_CATEGORY[name] for name in categories]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sim-To-Real Evidence Gap",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Enough for one-step transfer: `{report['enough_for_one_step_transfer']}`",
        f"- Transfer approved: `{report['transfer_approved']}`",
        f"- Hardware-approved checkpoints: `{report['hardware_approved_checkpoints']}`",
        "",
        "## Blocker Categories",
        "",
    ]
    if report["categories"]:
        for category, blockers in report["categories"].items():
            lines.append(f"- `{category}`: {', '.join(f'`{item}`' for item in blockers)}")
    else:
        lines.append("- `none`")
    lines.extend(["", "## Next Evidence To Collect", ""])
    lines.extend(f"- {item}" for item in report["action_items"] or ["No evidence gaps are currently reported."])
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
