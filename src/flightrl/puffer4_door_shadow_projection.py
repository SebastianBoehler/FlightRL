from __future__ import annotations

from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from flightrl.puffer4_door_contract import (
    DoorActionContract,
    FIXED_DOOR_LIVE_SAFETY_CONTRACT,
)
from flightrl.puffer4_door_shadow_identity import (
    EncodedShadowIdentity,
    SHADOW_IDENTITY_JSON_FIELD,
    SHADOW_IDENTITY_SHA256_FIELD,
)


SHADOW_PROJECTION_FIELDS = frozenset(
    {
        "policy_proposed_yawrate_deg_s",
        "yaw_only_projected_forward_m_s",
        "yaw_only_projected_yawrate_deg_s",
        "yaw_only_projection_saturated",
        "executed_previous_forward_normalized",
        "executed_previous_yaw_normalized",
    }
)


def project_fixed_door_shadow_row(
    row: Mapping,
    action_contract: DoorActionContract,
) -> dict:
    normalized_yaw = float(row["action_yaw"])
    if not isfinite(normalized_yaw):
        raise ValueError("shadow policy yaw proposal must be finite")
    proposed_yawrate = (
        normalized_yaw * action_contract.max_yawrate_deg_s
    )
    live_limit = FIXED_DOOR_LIVE_SAFETY_CONTRACT.max_yawrate_deg_s
    projected_yawrate = float(
        np.clip(proposed_yawrate, -live_limit, live_limit)
    )
    return dict(row) | {
        "policy_proposed_yawrate_deg_s": proposed_yawrate,
        "yaw_only_projected_forward_m_s": 0.0,
        "yaw_only_projected_yawrate_deg_s": projected_yawrate,
        "yaw_only_projection_saturated": (
            abs(proposed_yawrate) > live_limit
        ),
        "executed_previous_forward_normalized": 0.0,
        "executed_previous_yaw_normalized": 0.0,
    }


def bind_fixed_door_shadow_rows(
    rows: Sequence[Mapping],
    identity: EncodedShadowIdentity,
    action_contract: DoorActionContract,
) -> list[dict]:
    bound = []
    for row in rows:
        if (
            row.get("monitor_only") is not True
            or row.get("controls_drone") is not False
        ):
            raise ValueError("shadow evidence row must be non-actuating")
        projected = project_fixed_door_shadow_row(row, action_contract)
        projected[SHADOW_IDENTITY_JSON_FIELD] = identity.canonical_json
        projected[SHADOW_IDENTITY_SHA256_FIELD] = identity.sha256
        bound.append(projected)
    if not bound:
        raise ValueError("shadow evidence requires at least one row")
    return bound


def summarize_fixed_door_shadow_projection(
    rows: Sequence[Mapping],
    action_contract: DoorActionContract,
) -> dict:
    if not rows:
        raise ValueError("shadow projection requires evidence rows")
    forward = np.asarray(
        [float(row["yaw_only_projected_forward_m_s"]) for row in rows],
        dtype=np.float64,
    )
    yaw = np.asarray(
        [float(row["yaw_only_projected_yawrate_deg_s"]) for row in rows],
        dtype=np.float64,
    )
    previous = np.asarray(
        [
            (
                float(row["executed_previous_forward_normalized"]),
                float(row["executed_previous_yaw_normalized"]),
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    saturated = [
        bool(row["yaw_only_projection_saturated"]) for row in rows
    ]
    action_yaw = np.asarray(
        [float(row["action_yaw"]) for row in rows],
        dtype=np.float64,
    )
    proposed = np.asarray(
        [float(row["policy_proposed_yawrate_deg_s"]) for row in rows],
        dtype=np.float64,
    )
    limit = FIXED_DOOR_LIVE_SAFETY_CONTRACT.max_yawrate_deg_s
    expected_proposed = action_yaw * action_contract.max_yawrate_deg_s
    expected_yaw = np.clip(expected_proposed, -limit, limit)
    expected_saturated = np.abs(expected_proposed) > limit
    mapping_passed = bool(
        np.allclose(proposed, expected_proposed, rtol=0.0, atol=1.0e-9)
        and np.allclose(yaw, expected_yaw, rtol=0.0, atol=1.0e-9)
        and np.array_equal(
            np.asarray(saturated, dtype=bool),
            expected_saturated,
        )
    )
    finite = bool(
        np.isfinite(forward).all()
        and np.isfinite(yaw).all()
        and np.isfinite(previous).all()
    )
    forward_max = float(np.max(np.abs(forward)))
    yaw_max = float(np.max(np.abs(yaw)))
    previous_max = float(np.max(np.abs(previous)))
    passed = bool(
        finite
        and mapping_passed
        and forward_max == 0.0
        and yaw_max <= limit
        and previous_max == 0.0
    )
    return {
        "yaw_only_projection_contract": (
            FIXED_DOOR_LIVE_SAFETY_CONTRACT.to_report()
        ),
        "yaw_only_projection_contract_passed": passed,
        "yaw_only_projection_mapping_passed": mapping_passed,
        "yaw_only_projection_outputs_finite": finite,
        "yaw_only_projected_forward_abs_max_m_s": forward_max,
        "yaw_only_projected_abs_yawrate_max_deg_s": yaw_max,
        "yaw_only_projected_abs_yawrate_p95_deg_s": float(
            np.percentile(np.abs(yaw), 95)
        ),
        "yaw_only_projection_saturation_fraction": (
            sum(saturated) / len(saturated)
        ),
        "executed_previous_action_abs_max": previous_max,
    }
