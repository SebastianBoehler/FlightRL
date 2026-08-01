from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from flightrl.semantic.readiness_authority import (
    BOUNDED_FORWARD_AXIS_AUTHORITY,
    YAW_ONLY_AXIS_AUTHORITY,
    authority_decision_errors,
    bounded_forward_authority_fields,
    yaw_only_authority_fields,
)


def build_yaw_only_readiness(
    checkpoint: str | Path,
    sim_report_path: str | Path,
    replay_report_path: str | Path,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    sim_report_path = Path(sim_report_path).resolve()
    replay_report_path = Path(replay_report_path).resolve()
    sim = json.loads(sim_report_path.read_text())
    replay = json.loads(replay_report_path.read_text())

    sim_yaw_gate = bool(
        float(sim["collision_rate"]) == 0.0
        and float(sim["preacquisition_horizontal_p95_m_s"]) <= 0.03
        and float(sim["visible_yaw_sign_accuracy"]) >= 0.90
        and float(sim["visible_abs_yawrate_p95_deg_s"]) <= 10.1
        and float(sim["max_abs_yawrate_deg_s"]) <= 20.1
    )
    replay_yaw_gate = bool(replay["next_live_shadow_gate_passed"])
    passed = sim_yaw_gate and replay_yaw_gate
    return {
        "schema_version": 2,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "sim_report": str(sim_report_path),
        "sim_report_sha256": file_sha256(sim_report_path),
        "replay_report": str(replay_report_path),
        "replay_report_sha256": file_sha256(replay_report_path),
        **yaw_only_authority_fields(passed),
        "limits": {
            "search_abs_yawrate_deg_s": 20.0,
            "detected_abs_yawrate_deg_s": 8.0,
            "proposal_stale_s": 1.0,
        },
        "sim_yaw_gate_passed": sim_yaw_gate,
        "replay_yaw_gate_passed": replay_yaw_gate,
        "next_live_gate_passed": passed,
        "translation_authority_passed": False,
        "sim_mission_success_rate": float(sim["success_rate"]),
        "reason": (
            "Yaw-only live authority passed; learned translation remains disabled "
            "because the simulator mission-success gate did not pass."
            if passed
            else "Yaw-only authority remains disabled until both gates pass."
        ),
    }


def load_yaw_only_readiness(
    report_path: str | Path,
    checkpoint: str | Path,
) -> dict[str, Any]:
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    actual_sha256 = file_sha256(checkpoint)
    errors = []
    if report.get("schema_version") not in (1, 2):
        errors.append("unsupported readiness schema")
    if report.get("checkpoint_sha256") != actual_sha256:
        errors.append("checkpoint SHA-256 does not match the readiness report")
    for evidence_name in ("sim_report", "replay_report"):
        evidence_path = report.get(evidence_name)
        expected_hash = report.get(f"{evidence_name}_sha256")
        if not evidence_path or not Path(evidence_path).is_file():
            errors.append(f"{evidence_name} is missing")
        elif file_sha256(evidence_path) != expected_hash:
            errors.append(f"{evidence_name} SHA-256 does not match")
    errors.extend(
        authority_decision_errors(
            report,
            candidate_authority="yaw_only",
            candidate_axes=YAW_ONLY_AXIS_AUTHORITY,
        )
    )
    if report.get("translation_authority_passed") is not False:
        errors.append("readiness report must explicitly reject translation authority")
    if report.get("next_live_gate_passed") is not True:
        errors.append("next-live readiness gate did not pass")
    limits = report.get("limits", {})
    if float(limits.get("search_abs_yawrate_deg_s", float("inf"))) > 20.0:
        errors.append("search yaw limit exceeds 20 deg/s")
    if float(limits.get("detected_abs_yawrate_deg_s", float("inf"))) > 8.0:
        errors.append("detected-target yaw limit exceeds 8 deg/s")
    stale_s = float(limits.get("proposal_stale_s", float("inf")))
    if not 0.0 < stale_s <= 1.0:
        errors.append("proposal stale interval must be in (0, 1] seconds")
    if errors:
        raise ValueError(f"invalid readiness report {report_path}: {'; '.join(errors)}")
    return report


def build_bounded_forward_readiness(
    checkpoint: str | Path,
    training_report_path: str | Path,
    replay_report_path: str | Path,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    training_report_path = Path(training_report_path).resolve()
    replay_report_path = Path(replay_report_path).resolve()
    training = json.loads(training_report_path.read_text())
    replay = json.loads(replay_report_path.read_text())
    evaluation = training["evaluation"][training["selected_stage"]]["full"]
    sim_gate = bool(
        training.get("active_exploration")
        and training.get("shadow_gate_passed")
        and training.get("checkpoint_sha256") == file_sha256(checkpoint)
        and float(evaluation["success_rate"]) >= 0.50
        and float(evaluation["target_discovery_rate"]) >= 0.70
        and float(evaluation["collision_rate"]) <= 0.02
        and float(evaluation["unsafe_forward_fraction"]) <= 0.02
        and float(
            evaluation.get(
                "minimum_moving_navigation_clearance_m",
                evaluation.get(
                    "minimum_moving_horizontal_clearance_m",
                    evaluation["minimum_moving_front_clearance_m"],
                ),
            )
        )
        >= 0.25
        and float(evaluation["clearance_false_safe_fraction"]) <= 0.02
        and float(evaluation["max_lateral_vertical_action"]) <= 0.001
    )
    replay_gate = bool(replay.get("translation_shadow_gate_passed"))
    passed = sim_gate and replay_gate
    return {
        "schema_version": 2,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "training_report": str(training_report_path),
        "training_report_sha256": file_sha256(training_report_path),
        "replay_report": str(replay_report_path),
        "replay_report_sha256": file_sha256(replay_report_path),
        **bounded_forward_authority_fields(passed),
        "limits": {
            "max_forward_speed_m_s": 0.05,
            "search_abs_yawrate_deg_s": 15.0,
            "detected_abs_yawrate_deg_s": 8.0,
            "proposal_stale_s": 0.5,
            "max_authority_duration_s": 3.0,
            "max_displacement_m": 0.20,
            "minimum_predicted_clearance_m": 0.45,
            "maximum_predicted_collision_risk": 0.35,
        },
        "sim_translation_gate_passed": sim_gate,
        "replay_translation_gate_passed": replay_gate,
        "next_live_gate_passed": passed,
        "translation_authority_passed": passed,
        "reason": (
            "Bounded forward+yaw authority passed."
            if passed
            else "Bounded forward+yaw authority remains disabled until both gates pass."
        ),
    }


def load_bounded_forward_readiness(
    report_path: str | Path,
    checkpoint: str | Path,
) -> dict[str, Any]:
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    errors = _evidence_errors(
        report,
        checkpoint,
        ("training_report", "replay_report"),
    )
    errors.extend(
        authority_decision_errors(
            report,
            candidate_authority="bounded_forward_yaw",
            candidate_axes=BOUNDED_FORWARD_AXIS_AUTHORITY,
        )
    )
    if report.get("translation_authority_passed") is not True:
        errors.append("translation authority gate did not pass")
    if report.get("next_live_gate_passed") is not True:
        errors.append("next-live readiness gate did not pass")
    limits = report.get("limits", {})
    if float(limits.get("max_forward_speed_m_s", float("inf"))) > 0.05:
        errors.append("forward speed limit exceeds 0.05 m/s")
    if float(limits.get("max_authority_duration_s", float("inf"))) > 3.0:
        errors.append("authority duration exceeds 3 seconds")
    if float(limits.get("max_displacement_m", float("inf"))) > 0.20:
        errors.append("displacement limit exceeds 0.20 m")
    if errors:
        raise ValueError(f"invalid readiness report {report_path}: {'; '.join(errors)}")
    return report


def write_readiness(path: str | Path, report: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_errors(
    report: dict[str, Any],
    checkpoint: str | Path,
    evidence_names: tuple[str, ...],
) -> list[str]:
    errors = []
    if report.get("schema_version") not in (1, 2):
        errors.append("unsupported readiness schema")
    if report.get("checkpoint_sha256") != file_sha256(checkpoint):
        errors.append("checkpoint SHA-256 does not match the readiness report")
    for evidence_name in evidence_names:
        path = report.get(evidence_name)
        expected_hash = report.get(f"{evidence_name}_sha256")
        if not path or not Path(path).is_file():
            errors.append(f"{evidence_name} is missing")
        elif file_sha256(path) != expected_hash:
            errors.append(f"{evidence_name} SHA-256 does not match")
    return errors
