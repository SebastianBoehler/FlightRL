from __future__ import annotations

from math import hypot, isfinite
from numbers import Real
from typing import Any, Mapping

from flightrl.artifact_identity import bind_payload, require_bound_payload
from flightrl.puffer4_edge_action_contract import edge_action_contract_payload
from flightrl.puffer4_edge_wire import (
    EdgeTimingProfile,
    edge_timing_payload,
    edge_wire_contract,
)
from flightrl.puffer4_edge_schema import (
    EDGE_ACTION_DIM,
    EDGE_FRAME_PIXELS,
    EDGE_HEIGHT,
    EDGE_MISSION_TOKEN_COUNT,
    EDGE_OBSERVATION_DIM,
    EDGE_POLICY_CONTRACT_ID,
    EDGE_TARGET_VOCABULARY,
    EDGE_TELEMETRY_BOUNDS,
    EDGE_TELEMETRY_DIM,
    EDGE_WIDTH,
    TELEMETRY_SPECS,
)


def edge_target_id(target: str) -> int:
    try:
        return EDGE_TARGET_VOCABULARY.index(target)
    except ValueError as exc:
        raise ValueError(f"{target!r} is not an approved v3 target") from exc


def validate_edge_target_id(target_id: object) -> int:
    if type(target_id) is not int or not 0 <= target_id < EDGE_MISSION_TOKEN_COUNT:
        raise ValueError("AI Deck target ID is outside the approved v3 vocabulary")
    return target_id


def edge_target_one_hot(target_id: object) -> tuple[float, ...]:
    validated = validate_edge_target_id(target_id)
    return tuple(
        1.0 if index == validated else 0.0
        for index in range(EDGE_MISSION_TOKEN_COUNT)
    )


def edge_target_id_for_scene_object(
    object_id: object,
    bindings: Mapping[str, str],
) -> int:
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("semantic object ID must be a nonempty string")
    try:
        target = bindings[object_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"semantic object {object_id!r} has no explicit edge target binding"
        ) from exc
    return edge_target_id(target)


def validate_normalized_edge_telemetry(values: object) -> tuple[float, ...]:
    normalized = _validate_normalized_values(
        values,
        bounds=EDGE_TELEMETRY_BOUNDS,
        label="telemetry",
    )
    if abs(hypot(*normalized[6:9]) - 1.0) > 1.0e-4:
        raise ValueError("AI Deck body-up vector must have unit norm")
    if abs(hypot(*normalized[13:15]) - 1.0) > 1.0e-4:
        raise ValueError("AI Deck relative-yaw pair must have unit norm")
    return normalized


def validate_normalized_edge_action(values: object) -> tuple[float, ...]:
    return _validate_normalized_values(
        values,
        bounds=((-1.0, 1.0),) * EDGE_ACTION_DIM,
        label="action",
    )


def _validate_normalized_values(
    values: object,
    *,
    bounds: tuple[tuple[float, float], ...],
    label: str,
) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != len(bounds):
        raise ValueError(f"AI Deck normalized {label} has the wrong shape")
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AI Deck normalized {label} is not numeric") from exc
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(output)
        or not low <= output <= high
        for value, output, (low, high) in zip(values, normalized, bounds)
    ):
        raise ValueError(f"AI Deck normalized {label} is nonfinite or out of range")
    return normalized


def edge_policy_contract_report(
    *,
    hidden_size: int = 48,
    timing: EdgeTimingProfile | None = None,
) -> dict[str, Any]:
    return bind_payload(_payload(hidden_size, timing))


def verify_edge_policy_contract(
    report: Mapping[str, Any],
    *,
    hidden_size: int,
    timing: EdgeTimingProfile | None = None,
) -> None:
    payload = require_bound_payload(report, label="AI Deck policy contract")
    if payload != _payload(hidden_size, timing):
        raise ValueError("AI Deck policy contract is not an approved v3 payload")


def _payload(
    hidden_size: int,
    timing: EdgeTimingProfile | None,
) -> dict[str, Any]:
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
    ):
        raise ValueError("AI Deck recurrent hidden size must be an integer")
    if not 32 <= hidden_size <= 64:
        raise ValueError("AI Deck recurrent hidden size must be in [32, 64]")
    frame_end = EDGE_FRAME_PIXELS
    telemetry_end = frame_end + EDGE_TELEMETRY_DIM
    return {
        "contract_id": EDGE_POLICY_CONTRACT_ID,
        "schema_version": 3,
        "runtime": {
            "training": "mac_edge_shaped_pytorch_reference",
            "inference": "gap8_int8_unimplemented_target",
            "exact_deployment_graph_available": False,
            "timing_bound": timing is not None,
            "exact_training_adapter_available": True,
            "exact_training_authority": True,
            "exact_training_target_ids": [edge_target_id("door")],
            "safety_owner": "stm32",
            "timing": edge_timing_payload(timing),
        },
        "observation": {
            "flat_values": EDGE_OBSERVATION_DIM,
            "flat_dtype": "float32",
            "segments": {
                "current_gray4": [0, frame_end],
                "telemetry": [frame_end, telemetry_end],
                "mission_one_hot": [telemetry_end, EDGE_OBSERVATION_DIM],
            },
            "frame": {
                "width": EDGE_WIDTH,
                "height": EDGE_HEIGHT,
                "pixels": EDGE_FRAME_PIXELS,
                "history_planes": 1,
                "wire_encoding": "packed_gray4_row_major",
                "nibble_order": "even_pixel_high_odd_pixel_low",
                "packed_gray4_bytes": EDGE_FRAME_PIXELS // 2,
                "model_formula": "float32(unpacked_nibble) / 15.0",
                "model_values": [index / 15.0 for index in range(16)],
                "incomplete_frame": "reject_and_require_next_state_reset",
            },
            "telemetry": {
                "values": EDGE_TELEMETRY_DIM,
                "wire_dtype": "float32_le_normalized",
                "order": [spec[0] for spec in TELEMETRY_SPECS],
                "fields": [_telemetry_field(spec) for spec in TELEMETRY_SPECS],
            },
            "mission_token": {
                "wire_type": "uint8_target_id",
                "count": EDGE_MISSION_TOKEN_COUNT,
                "vocabulary": {
                    str(index): target
                    for index, target in enumerate(EDGE_TARGET_VOCABULARY)
                },
                "model_encoding": "float32_one_hot_at_target_id",
                "invalid_id": "reject_packet",
                "open_vocabulary_allowed": False,
            },
            "previous_action": {
                "meaning": "last_setpoint_actually_applied_by_stm32_after_safety",
                "frame": "the_applied_setpoints_own_step_frames",
                "initial_value": [0.0, 0.0, 0.0, 0.0],
                "proposal_is_not_feedback": True,
            },
            "host_detector_required": False,
            "host_phase_required": False,
        },
        "model": {
            "status": "edge_shaped_executable_pytorch_reference",
            "visual_trunks": 1,
            "grounding_head": ["visible", "center_x", "center_y", "scale"],
            "grounding_condition": (
                "active_target_one_hot_multiplicatively_gates_visual_features"
            ),
            "grounding_label_semantics": {
                "visible": "1 iff active target has a labeled box; otherwise 0",
                "center_x": "(x_min + x_max) / (width - 1) - 1; image right positive",
                "center_y": "(y_min + y_max) / (height - 1) - 1; image down positive",
                "scale": "sqrt(((x_max-x_min+1)*(y_max-y_min+1))/(width*height))",
                "absent_target": "visible label zero and supervised; box labels zero and losses are masked",
            },
            "grounding_features_shared_with_actor": True,
            "actor_grounding": "center and scale multiplied by predicted visibility",
            "recurrent_kind": "hard_gated_relu6",
            "hidden_size": hidden_size,
            "hidden_state_range": [0.0, 6.0],
            "parameter_limit": 50_000,
            "quantized_parameter_byte_limit": 64 * 1024,
            "training_only_outputs_stripped": ["critic", "logstd"],
        },
        "recurrent_reset": {
            "wire_flag": "policy_input.flags_bit_0",
            "operation": "zero_hidden_before_inference",
            "required_on": [
                "actor_boot",
                "mission_start_or_target_id_change",
                "arming_epoch_change",
                "estimator_or_mission_origin_reset",
                "invalid_incomplete_duplicate_or_reordered_input",
                "capture_period_above_bound_maximum_or_dropped_frame",
            ],
            "state_commit": "only_after_valid_packet_and_successful_inference",
            "rejected_packet": "does_not_update_state_and_next_valid_packet_resets",
        },
        "action": edge_action_contract_payload(),
        "mission_boundary": {
            "policy_owned_phases": ["search", "navigate"],
            "stm32_owned_phases": [
                "preflight", "takeoff", "recover", "hold", "land", "abort"
            ],
            "goal_reached_owner": "stm32_mission_supervisor",
            "policy_may_assert_goal_reached": False,
            "target_binding": "explicit semantic object ID to approved target ID; scene index forbidden",
            "epoch_rule": "mission or arming epoch change requires recurrent reset",
        },
        "wire": edge_wire_contract(),
        "promotion_requires": [
            "measured_runtime_timing_profile_bound",
            "freeze_preprocessing_operator_and_quantization_manifest",
            "pytorch_float_vs_float_c_recurrent_sequence_max_abs_1e-5",
            "int8_calibration_and_heldout_mission_regression_gate",
            "host_int8_c_vs_gap8_bit_exact_all_outputs_and_state",
            "gap8_elf_memory_and_latency_measurements",
            "cpx_echo_sequence_epoch_freshness_and_stm32_deadman_tests",
        ],
    }


def _telemetry_field(spec: tuple[Any, ...]) -> dict[str, Any]:
    name, source_unit, scale, clip, reference_frame, formula = spec
    return {
        "name": name,
        "source_unit": source_unit,
        "scale": scale,
        "clip": list(clip),
        "reference_frame": reference_frame,
        "formula": f"clip({formula}, {clip[0]}, {clip[1]})",
    }
