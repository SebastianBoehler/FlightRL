from __future__ import annotations

import json

import pytest

from flightrl.puffer4_door_contract import CORRECTED_DOOR_ACTION_CONTRACT
from flightrl.puffer4_door_shadow_io import detection_yaw_alignment
from flightrl.puffer4_door_shadow_projection import (
    project_fixed_door_shadow_row,
    summarize_fixed_door_shadow_projection,
)


def test_shadow_projection_is_hypothetical_yaw_only_at_eight_degrees() -> None:
    projected = project_fixed_door_shadow_row(
        {"action_forward": 0.6, "action_yaw": 0.5},
        CORRECTED_DOOR_ACTION_CONTRACT,
    )

    assert projected["policy_proposed_yawrate_deg_s"] == 35.0
    assert projected["yaw_only_projected_forward_m_s"] == 0.0
    assert projected["yaw_only_projected_yawrate_deg_s"] == 8.0
    assert projected["yaw_only_projection_saturated"] is True
    assert projected["executed_previous_forward_normalized"] == 0.0
    assert projected["executed_previous_yaw_normalized"] == 0.0


def test_shadow_projection_summary_proves_zero_translation_and_history() -> None:
    rows = [
        project_fixed_door_shadow_row(
            {"action_forward": 0.4, "action_yaw": yaw},
            CORRECTED_DOOR_ACTION_CONTRACT,
        )
        for yaw in (-0.5, 0.0, 0.5)
    ]

    summary = summarize_fixed_door_shadow_projection(
        rows,
        CORRECTED_DOOR_ACTION_CONTRACT,
    )

    assert summary["yaw_only_projection_contract_passed"] is True
    assert summary["yaw_only_projection_mapping_passed"] is True
    assert summary["yaw_only_projected_forward_abs_max_m_s"] == 0.0
    assert summary["yaw_only_projected_abs_yawrate_max_deg_s"] == 8.0
    assert summary["executed_previous_action_abs_max"] == 0.0
    assert summary["yaw_only_projection_saturation_fraction"] == pytest.approx(
        2 / 3
    )


def test_shadow_projection_rejects_bounded_but_relabelled_yaw() -> None:
    row = project_fixed_door_shadow_row(
        {"action_forward": 0.4, "action_yaw": 0.5},
        CORRECTED_DOOR_ACTION_CONTRACT,
    )
    row["yaw_only_projected_yawrate_deg_s"] = 7.0

    summary = summarize_fixed_door_shadow_projection(
        [row],
        CORRECTED_DOOR_ACTION_CONTRACT,
    )

    assert summary["yaw_only_projection_mapping_passed"] is False
    assert summary["yaw_only_projection_contract_passed"] is False


def test_off_center_detection_with_zero_projected_yaw_is_not_aligned() -> None:
    detection = json.dumps(
        {"box": {"x_min": 0.0, "x_max": 0.2}}
    )
    row = project_fixed_door_shadow_row(
        {
            "detection": detection,
            "action_forward": 0.0,
            "action_yaw": 0.0,
        },
        CORRECTED_DOOR_ACTION_CONTRACT,
    )

    samples, accuracy = detection_yaw_alignment(
        [row],
        yaw_field="yaw_only_projected_yawrate_deg_s",
    )

    assert samples == 1
    assert accuracy == 0.0
