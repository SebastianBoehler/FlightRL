from __future__ import annotations

from math import isclose
from typing import Any

from flightrl.puffer4_door_bundle import FixedDoorCheckpointBundle
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
    approved_door_evidence_age_contract_from_report,
)
from flightrl.puffer4_door_live_evidence import FixedDoorLiveEvidence
from flightrl.puffer4_door_shadow_identity import (
    SHADOW_IDENTITY_JSON_FIELD,
    SHADOW_IDENTITY_SHA256_FIELD,
    decode_fixed_door_shadow_identity,
    require_shadow_identity_matches_evidence,
)


def action_contract_matches(
    bundle: FixedDoorCheckpointBundle,
    shadow: dict,
) -> bool:
    contract = bundle.action_contract
    return bool(
        shadow.get("action_contract_id") == contract.contract_id
        and shadow.get("action_contract_sha256") == contract.sha256()
    )


def policy_contract_matches(
    bundle: FixedDoorCheckpointBundle,
    shadow: dict,
) -> bool:
    encoded = bundle.policy_contract
    return bool(
        shadow.get("policy_contract_id") == encoded.get("contract_id")
        and shadow.get("policy_contract_sha256") == encoded.get("sha256")
    )


def evidence_age_contract_matches(shadow: dict) -> bool:
    encoded = shadow.get("evidence_age_runtime_contract")
    if not isinstance(encoded, dict):
        return False
    try:
        decoded = approved_door_evidence_age_contract_from_report(encoded)
    except (TypeError, ValueError):
        return False
    return decoded == FIXED_DOOR_EVIDENCE_AGE_CONTRACT


def shadow_run_identity_matches(
    evidence: FixedDoorLiveEvidence,
    summary: dict,
    csv_evidence: dict,
) -> bool:
    try:
        identity = decode_fixed_door_shadow_identity(
            csv_evidence[SHADOW_IDENTITY_JSON_FIELD],
            csv_evidence[SHADOW_IDENTITY_SHA256_FIELD],
        )
        require_shadow_identity_matches_evidence(identity, evidence)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        summary.get("shadow_run_identity") == identity.payload
        and summary.get(SHADOW_IDENTITY_JSON_FIELD)
        == identity.canonical_json
        and summary.get(SHADOW_IDENTITY_SHA256_FIELD) == identity.sha256
    )


def shadow_projection_matches(summary: dict, csv_evidence: dict) -> bool:
    exact = (
        "yaw_only_projection_contract",
        "yaw_only_projection_contract_passed",
        "yaw_only_projection_mapping_passed",
        "yaw_only_projection_outputs_finite",
    )
    numeric = (
        "yaw_only_projected_forward_abs_max_m_s",
        "yaw_only_projected_abs_yawrate_max_deg_s",
        "yaw_only_projected_abs_yawrate_p95_deg_s",
        "yaw_only_projection_saturation_fraction",
        "executed_previous_action_abs_max",
        "yaw_only_detection_yaw_sign_accuracy",
    )
    return bool(
        all(summary.get(key) == csv_evidence.get(key) for key in exact)
        and all(
            same_optional_number(summary.get(key), csv_evidence.get(key))
            for key in numeric
        )
        and summary.get("yaw_only_detection_yaw_alignment_samples")
        == csv_evidence.get("yaw_only_detection_yaw_alignment_samples")
        and csv_evidence.get("yaw_only_projection_contract_passed") is True
    )


def summary_matches_csv(summary: dict, evidence: dict) -> bool:
    exact_keys = (
        "rows",
        "controls_drone",
        "monitor_only",
        "phase_counts",
        "detection_rows",
        "detection_yaw_alignment_samples",
        "all_actions_finite",
        "timestamps_strictly_increasing",
        "frame_width",
        "frame_height",
        "frame_geometry_contract_passed",
        "phase_coverage_passed",
        "frame_indices_strictly_increasing",
        "frame_index_gap_count",
        "stream_dropped_frames",
        "stream_drop_counter_consistent",
        "grounding_result_frame_order_passed",
        "grounding_unique_results",
        "shadow_run_identity",
        SHADOW_IDENTITY_JSON_FIELD,
        SHADOW_IDENTITY_SHA256_FIELD,
    )
    if any(summary.get(key) != evidence.get(key) for key in exact_keys):
        return False
    numeric_keys = (
        "detection_yaw_sign_accuracy",
        "inference_ms_p95",
        "grounding_age_s_p95",
        "sampled_coverage_s",
        "grounding_result_rate_hz",
        "grounding_update_interval_s_p95",
        "grounding_inference_ms_p95",
        "grounding_age_margin_s_p05",
    )
    return all(
        same_optional_number(summary.get(key), evidence.get(key))
        for key in numeric_keys
    ) and shadow_projection_matches(summary, evidence)


def same_optional_number(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return isclose(
        float(left),
        float(right),
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    )
