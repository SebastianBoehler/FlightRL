from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np

from flightrl.puffer4_door_contract import APPROVED_DOOR_ACTION_CONTRACTS
from flightrl.puffer4_door_shadow_identity import (
    EncodedShadowIdentity,
    SHADOW_IDENTITY_JSON_FIELD,
    SHADOW_IDENTITY_SHA256_FIELD,
    decode_fixed_door_shadow_identity,
)
from flightrl.puffer4_door_shadow_metrics import (
    detection_yaw_alignment,
    shadow_capture_contract,
    shadow_stream_metrics,
)
from flightrl.puffer4_door_shadow_projection import (
    SHADOW_PROJECTION_FIELDS,
    summarize_fixed_door_shadow_projection,
)


SHADOW_CSV_REQUIRED_FIELDS = frozenset(
    {
        "action_forward",
        "action_yaw",
        "controls_drone",
        "detection",
        "frame_index",
        "frame_height",
        "frame_host_time_s",
        "frame_width",
        "grounding_age_s",
        "grounding_inference_ms",
        "grounding_result_frame_index",
        "inference_ms",
        "monitor_only",
        "phase",
        "target_detected",
        "stream_dropped_frames",
        SHADOW_IDENTITY_JSON_FIELD,
        SHADOW_IDENTITY_SHA256_FIELD,
    }
) | SHADOW_PROJECTION_FIELDS


def shadow_identity_from_rows(
    rows: list[Mapping],
) -> EncodedShadowIdentity:
    if not rows:
        raise ValueError("shadow evidence contains no rows")
    try:
        identities = [
            decode_fixed_door_shadow_identity(
                str(row[SHADOW_IDENTITY_JSON_FIELD]),
                str(row[SHADOW_IDENTITY_SHA256_FIELD]),
            )
            for row in rows
        ]
    except KeyError as exc:
        raise ValueError(
            f"shadow evidence is missing identity field: {exc.args[0]}"
        ) from exc
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("shadow CSV contains mixed run identities")
    return identities[0]


def read_shadow_csv_evidence(path: str | Path) -> dict:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(
            SHADOW_CSV_REQUIRED_FIELDS - set(reader.fieldnames or ())
        )
        if missing:
            raise ValueError(f"shadow CSV missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("shadow CSV contains no evidence rows")
    identity = shadow_identity_from_rows(rows)
    action_identity = identity.payload["action_contract"]
    action_contract = APPROVED_DOOR_ACTION_CONTRACTS.get(
        action_identity["contract_id"]
    )
    if (
        action_contract is None
        or action_contract.sha256() != action_identity["sha256"]
    ):
        raise ValueError("shadow identity action contract is not approved")
    normalized = [_normalize_row(row) for row in rows]
    phases = {
        phase: sum(row["phase"] == phase for row in normalized)
        for phase in ("search", "track", "approach", "recover")
    }
    forward = np.asarray([row["action_forward"] for row in normalized])
    yaw = np.asarray([row["action_yaw"] for row in normalized])
    grounding_ages = [
        row["grounding_age_s"]
        for row in normalized
        if row["grounding_age_s"] is not None
    ]
    alignment_samples, alignment = detection_yaw_alignment(normalized)
    projection_samples, projection_alignment = detection_yaw_alignment(
        normalized,
        yaw_field="yaw_only_projected_yawrate_deg_s",
    )
    return {
        "shadow_run_identity": dict(identity.payload),
        SHADOW_IDENTITY_JSON_FIELD: identity.canonical_json,
        SHADOW_IDENTITY_SHA256_FIELD: identity.sha256,
        "rows": len(normalized),
        "controls_drone": any(row["controls_drone"] for row in normalized),
        "monitor_only": all(row["monitor_only"] for row in normalized),
        "phase_counts": phases,
        "detection_rows": sum(row["target_detected"] for row in normalized),
        "detection_yaw_alignment_samples": alignment_samples,
        "detection_yaw_sign_accuracy": alignment,
        "yaw_only_detection_yaw_alignment_samples": projection_samples,
        "yaw_only_detection_yaw_sign_accuracy": projection_alignment,
        "all_actions_finite": bool(
            np.isfinite(forward).all() and np.isfinite(yaw).all()
        ),
        "inference_ms_p95": float(
            np.percentile([row["inference_ms"] for row in normalized], 95)
        ),
        "grounding_age_s_p95": (
            None
            if not grounding_ages
            else float(np.percentile(grounding_ages, 95))
        ),
        **shadow_capture_contract(normalized, phases),
        **shadow_stream_metrics(normalized),
        **summarize_fixed_door_shadow_projection(
            normalized,
            action_contract,
        ),
    }


def _normalize_row(row: dict[str, str]) -> dict:
    phase = row["phase"]
    if phase not in ("search", "track", "approach", "recover"):
        raise ValueError(f"shadow CSV has unknown phase: {phase!r}")
    return row | {
        "action_forward": float(row["action_forward"]),
        "action_yaw": float(row["action_yaw"]),
        "policy_proposed_yawrate_deg_s": float(
            row["policy_proposed_yawrate_deg_s"]
        ),
        "yaw_only_projected_forward_m_s": float(
            row["yaw_only_projected_forward_m_s"]
        ),
        "yaw_only_projected_yawrate_deg_s": float(
            row["yaw_only_projected_yawrate_deg_s"]
        ),
        "yaw_only_projection_saturated": _csv_bool(
            row["yaw_only_projection_saturated"]
        ),
        "executed_previous_forward_normalized": float(
            row["executed_previous_forward_normalized"]
        ),
        "executed_previous_yaw_normalized": float(
            row["executed_previous_yaw_normalized"]
        ),
        "controls_drone": _csv_bool(row["controls_drone"]),
        "frame_index": int(row["frame_index"]),
        "frame_height": int(row["frame_height"]),
        "frame_host_time_s": float(row["frame_host_time_s"]),
        "frame_width": int(row["frame_width"]),
        "grounding_age_s": (
            None
            if row["grounding_age_s"] == ""
            else float(row["grounding_age_s"])
        ),
        "grounding_inference_ms": float(row["grounding_inference_ms"]),
        "grounding_result_frame_index": int(
            row["grounding_result_frame_index"]
        ),
        "inference_ms": float(row["inference_ms"]),
        "monitor_only": _csv_bool(row["monitor_only"]),
        "target_detected": _csv_bool(row["target_detected"]),
        "stream_dropped_frames": int(row["stream_dropped_frames"]),
    }


def _csv_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"shadow CSV has invalid boolean: {value!r}")
