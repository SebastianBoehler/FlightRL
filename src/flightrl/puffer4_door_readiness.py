from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightrl.puffer4_door_contract import (
    DoorLiveSafetyContract,
    FIXED_DOOR_LIVE_SAFETY_CONTRACT,
)
from flightrl.puffer4_door_live_evidence import (
    validate_fixed_door_live_evidence,
)
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
    approved_door_evidence_age_contract_from_report,
)
from flightrl.puffer4_door_shadow_io import (
    MIN_SAMPLED_COVERAGE_S,
    read_shadow_csv_evidence,
)
from flightrl.puffer4_door_readiness_evidence import (
    action_contract_matches,
    evidence_age_contract_matches,
    policy_contract_matches,
    shadow_projection_matches,
    shadow_run_identity_matches,
    summary_matches_csv,
)
from flightrl.semantic.readiness import (
    file_sha256,
    load_yaw_only_readiness,
    yaw_only_authority_fields,
)

MAX_SHADOW_DETECTOR_INFERENCE_MS_P95 = 750.0
MIN_SHADOW_GROUNDING_RESULT_RATE_HZ = 1.0
MIN_SHADOW_GROUNDING_AGE_MARGIN_S = 0.25


def build_fixed_door_yaw_readiness(
    checkpoint: str | Path,
    simulation_report_path: str | Path,
    shadow_summary_path: str | Path,
    shadow_csv_path: str | Path,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    simulation_report_path = Path(simulation_report_path).resolve()
    shadow_summary_path = Path(shadow_summary_path).resolve()
    shadow_csv_path = Path(shadow_csv_path).resolve()
    live_evidence = validate_fixed_door_live_evidence(
        checkpoint,
        simulation_report_path,
    )
    if live_evidence.kind != "promotion_v3":
        raise ValueError(
            "fixed-door yaw readiness requires promotion.v3 evidence"
        )
    bundle = live_evidence.bundle
    simulation = bundle.raw_report
    shadow = json.loads(shadow_summary_path.read_text())
    digest = bundle.checkpoint_sha256
    simulation_digest = bundle.report_sha256
    csv_evidence = read_shadow_csv_evidence(shadow_csv_path)
    full = simulation["full_camera"]
    masked = simulation["masked_camera"]
    live_yaw_cap = simulation["live_yaw_cap_challenge"]["metrics"]
    default_camera_passed = bool(
        float(full["success_rate"]) >= 0.70
        and float(full["outside_fov_success_rate"]) >= 0.65
        and float(full["collision_rate"]) <= 0.02
        and float(masked["success_rate"]) <= 0.10
        and float(full["success_rate"]) - float(masked["success_rate"]) >= 0.50
    )
    live_yaw_cap_passed = bool(
        float(live_yaw_cap["success_rate"]) >= 0.70
        and float(live_yaw_cap["outside_fov_success_rate"]) >= 0.65
        and float(live_yaw_cap["collision_rate"]) <= 0.03
    )
    sim_passed = default_camera_passed and live_yaw_cap_passed
    report_binding_passed = bool(
        shadow.get("checkpoint") == str(checkpoint)
        and shadow.get("checkpoint_sha256") == digest
        and shadow.get("evaluation_report") == str(simulation_report_path)
        and shadow.get("evaluation_report_sha256") == simulation_digest
    )
    action_binding_passed = action_contract_matches(bundle, shadow)
    policy_binding_passed = policy_contract_matches(bundle, shadow)
    evidence_age_binding_passed = evidence_age_contract_matches(shadow)
    csv_binding_passed = summary_matches_csv(shadow, csv_evidence)
    identity_binding_passed = shadow_run_identity_matches(
        live_evidence,
        shadow,
        csv_evidence,
    )
    projection_binding_passed = shadow_projection_matches(
        shadow,
        csv_evidence,
    )
    shadow_passed = bool(
        shadow.get("checkpoint") == str(checkpoint)
        and shadow.get("controls_drone") is False
        and shadow.get("monitor_only") is True
        and report_binding_passed
        and action_binding_passed
        and policy_binding_passed
        and evidence_age_binding_passed
        and csv_binding_passed
        and identity_binding_passed
        and projection_binding_passed
        and csv_evidence["controls_drone"] is False
        and csv_evidence["monitor_only"] is True
        and csv_evidence["all_actions_finite"] is True
        and int(csv_evidence["rows"]) >= 50
        and float(csv_evidence["sampled_coverage_s"] or 0.0)
        >= MIN_SAMPLED_COVERAGE_S
        and csv_evidence["timestamps_strictly_increasing"] is True
        and csv_evidence["frame_geometry_contract_passed"] is True
        and csv_evidence["phase_coverage_passed"] is True
        and int(csv_evidence["detection_rows"]) >= 20
        and int(csv_evidence["detection_yaw_alignment_samples"]) >= 10
        and float(csv_evidence["detection_yaw_sign_accuracy"] or 0.0) >= 0.75
        and int(
            csv_evidence["yaw_only_detection_yaw_alignment_samples"]
        )
        >= 10
        and float(
            csv_evidence["yaw_only_detection_yaw_sign_accuracy"] or 0.0
        )
        >= 0.75
        and csv_evidence["frame_indices_strictly_increasing"] is True
        and csv_evidence["stream_drop_counter_consistent"] is True
        and _at_most(csv_evidence.get("stream_dropped_frames"), 5)
        and csv_evidence["grounding_result_frame_order_passed"] is True
        and float(csv_evidence["grounding_result_rate_hz"])
        >= MIN_SHADOW_GROUNDING_RESULT_RATE_HZ
        and float(csv_evidence["grounding_inference_ms_p95"])
        <= MAX_SHADOW_DETECTOR_INFERENCE_MS_P95
        and float(csv_evidence["grounding_age_margin_s_p05"] or -1.0)
        >= MIN_SHADOW_GROUNDING_AGE_MARGIN_S
        and float(csv_evidence["inference_ms_p95"]) <= 10.0
        and _at_most(
            csv_evidence["grounding_age_s_p95"],
            FIXED_DOOR_EVIDENCE_AGE_CONTRACT.maximum_evidence_age_s,
        )
    )
    passed = sim_passed and shadow_passed
    live_safety = FIXED_DOOR_LIVE_SAFETY_CONTRACT
    return {
        "schema_version": 2,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "sim_report": str(simulation_report_path),
        "sim_report_sha256": simulation_digest,
        "evaluation_report": str(simulation_report_path),
        "evaluation_report_sha256": simulation_digest,
        "lineage_report": str(bundle.lineage_report_path),
        "lineage_report_sha256": bundle.lineage_report_sha256,
        "replay_report": str(shadow_summary_path),
        "replay_report_sha256": file_sha256(shadow_summary_path),
        "shadow_csv": str(shadow_csv_path),
        "shadow_csv_sha256": file_sha256(shadow_csv_path),
        **yaw_only_authority_fields(passed),
        "live_safety_contract": live_safety.to_report(),
        "evidence_age_runtime_contract": FIXED_DOOR_EVIDENCE_AGE_CONTRACT.to_report(),
        "limits": live_safety.readiness_limits(),
        "sim_yaw_gate_passed": sim_passed,
        "sim_default_camera_gate_passed": default_camera_passed,
        "sim_live_yaw_cap_gate_passed": live_yaw_cap_passed,
        "replay_yaw_gate_passed": shadow_passed,
        "shadow_report_binding_passed": report_binding_passed,
        "shadow_action_contract_binding_passed": action_binding_passed,
        "shadow_policy_contract_binding_passed": policy_binding_passed,
        "shadow_evidence_age_binding_passed": evidence_age_binding_passed,
        "shadow_csv_binding_passed": csv_binding_passed,
        "shadow_run_identity_binding_passed": identity_binding_passed,
        "shadow_yaw_only_projection_binding_passed": (
            projection_binding_passed
        ),
        "shadow_sampled_coverage_s": csv_evidence["sampled_coverage_s"],
        "shadow_frame_width": csv_evidence["frame_width"],
        "shadow_frame_height": csv_evidence["frame_height"],
        "shadow_phase_counts": csv_evidence["phase_counts"],
        "shadow_stream_dropped_frames": csv_evidence[
            "stream_dropped_frames"
        ],
        "shadow_frame_index_gap_count": csv_evidence[
            "frame_index_gap_count"
        ],
        "shadow_grounding_inference_ms_p95": csv_evidence[
            "grounding_inference_ms_p95"
        ],
        "shadow_grounding_result_rate_hz": csv_evidence[
            "grounding_result_rate_hz"
        ],
        "shadow_grounding_age_margin_s_p05": csv_evidence[
            "grounding_age_margin_s_p05"
        ],
        "next_live_gate_passed": passed,
        "translation_authority_passed": False,
        "sim_mission_success_rate": float(full["success_rate"]),
        "sim_outside_fov_success_rate": float(
            full["outside_fov_success_rate"]
        ),
        "sim_collision_rate": float(full["collision_rate"]),
        "masked_camera_success_rate": float(masked["success_rate"]),
        "live_yaw_cap_success_rate": float(live_yaw_cap["success_rate"]),
        "live_yaw_cap_outside_fov_success_rate": float(
            live_yaw_cap["outside_fov_success_rate"]
        ),
        "live_yaw_cap_collision_rate": float(
            live_yaw_cap["collision_rate"]
        ),
        "reason": (
            "Fixed-door student is ready for an initial bounded yaw-only flight."
            if passed
            else "Yaw-only authority remains disabled until simulation and real shadow evidence pass."
        ),
    }


def load_fixed_door_yaw_readiness(
    report_path: str | Path,
    checkpoint: str | Path,
    evaluation_report: str | Path,
) -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    evaluation_report = Path(evaluation_report).resolve()
    report = load_yaw_only_readiness(report_path, checkpoint)
    errors = []
    if report.get("readiness_report") != str(report_path):
        errors.append("readiness report path does not match")
    if report.get("checkpoint") != str(checkpoint):
        errors.append("checkpoint path does not match")
    if report.get("evaluation_report") != str(evaluation_report):
        errors.append("evaluation report path does not match")
    elif report.get("evaluation_report_sha256") != file_sha256(
        evaluation_report
    ):
        errors.append("evaluation report SHA-256 does not match")
    shadow_csv = report.get("shadow_csv")
    if not shadow_csv or not Path(shadow_csv).is_file():
        errors.append("shadow CSV is missing")
    elif file_sha256(shadow_csv) != report.get("shadow_csv_sha256"):
        errors.append("shadow CSV SHA-256 does not match")
    encoded_safety = report.get("live_safety_contract")
    try:
        loaded_safety = DoorLiveSafetyContract.from_report(encoded_safety or {})
    except (TypeError, ValueError):
        errors.append("fixed-door live safety contract is invalid")
    else:
        if loaded_safety != FIXED_DOOR_LIVE_SAFETY_CONTRACT:
            errors.append("fixed-door live safety contract does not match")
    encoded_evidence_age = report.get("evidence_age_runtime_contract")
    try:
        if not isinstance(encoded_evidence_age, dict):
            raise ValueError
        loaded_evidence_age = (
            approved_door_evidence_age_contract_from_report(
                encoded_evidence_age
            )
        )
    except (TypeError, ValueError):
        errors.append("fixed-door evidence-age runtime contract is invalid")
    else:
        if loaded_evidence_age != FIXED_DOOR_EVIDENCE_AGE_CONTRACT:
            errors.append(
                "fixed-door evidence-age runtime contract does not match"
            )
    limits = report["limits"]
    if limits != FIXED_DOOR_LIVE_SAFETY_CONTRACT.readiness_limits():
        errors.append("fixed-door readiness limits do not match live safety contract")
    if not errors:
        rebuilt = build_fixed_door_yaw_readiness(
            checkpoint,
            evaluation_report,
            report["replay_report"],
            report["shadow_csv"],
        )
        if rebuilt["next_live_gate_passed"] is not True:
            errors.append("fixed-door evidence no longer passes")
    if errors:
        raise ValueError(
            f"invalid fixed-door readiness report {report_path}: "
            + "; ".join(errors)
        )
    return report


def bind_fixed_door_readiness_identity(
    report: dict[str, Any],
    report_path: str | Path,
) -> dict[str, Any]:
    return report | {"readiness_report": str(Path(report_path).resolve())}

def _at_most(value: Any, maximum: float) -> bool:
    return value is not None and float(value) <= maximum
