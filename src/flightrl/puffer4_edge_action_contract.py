from __future__ import annotations

from typing import Any

from flightrl.puffer4_edge_schema import ACTION_SPECS


EDGE_CONTROLLED_ACTION_AXES = ("vx", "yaw_rate")
EDGE_CONTROLLED_TELEMETRY_INDICES = (15, 18)
EDGE_STRUCTURALLY_ZERO_ACTION_AXES = ("vy", "vz")


def edge_action_contract_payload() -> dict[str, Any]:
    return {
        "wire_type": "float32_le_normalized_proposal",
        "order": [spec[0] for spec in ACTION_SPECS],
        "controlled_axes": list(EDGE_CONTROLLED_ACTION_AXES),
        "structurally_zero_axes": list(EDGE_STRUCTURALLY_ZERO_ACTION_AXES),
        "parameterization": {
            "kind": "bounded_residual_over_stm32_applied_previous_action",
            "feedback_telemetry_indices": list(
                EDGE_CONTROLLED_TELEMETRY_INDICES
            ),
            "feedback_index_space": "zero_based_telemetry_segment",
            "learned_delta_clip": [-1.0, 1.0],
            "final_clip": [-1.0, 1.0],
            "delta_head_initialization": "exact_zero_weights_and_bias",
            "initial_policy": "controlled_axis_persistence",
        },
        "normalized_clip": [-1.0, 1.0],
        "physical_mapping": [_action_field(spec) for spec in ACTION_SPECS],
        "consumer": "stm32_safety_envelope",
        "freshness_required": True,
        "feedback": "next_observation_contains_stm32_applied_setpoint",
    }


def _action_field(spec: tuple[Any, ...]) -> dict[str, Any]:
    name, unit, scale, reference_frame = spec
    return {
        "name": name,
        "unit": unit,
        "scale": scale,
        "reference_frame": reference_frame,
        "formula": f"clip(normalized_{name}, -1, 1) * {scale}",
    }
