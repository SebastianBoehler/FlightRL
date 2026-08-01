from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

from flightrl.puffer4_door_evidence import DOOR_EVIDENCE_DIM
from flightrl.puffer4_door_observation import DOOR_PHASE_DIM, DOOR_SENSOR_DIM
from flightrl.puffer4_door_policy import (
    DOOR_HEIGHT,
    DOOR_OBS_DIM,
    DOOR_PIXELS,
    DOOR_POLICY_OBS_DIM,
    DOOR_PRIVILEGED_DIM,
    DOOR_WIDTH,
)


_V1_IDENTITY = ("fixed-door-recurrent-policy-v1", 1)
_V2_IDENTITY = ("fixed-door-recurrent-policy-v2", 2)
_APPROVED_SELF_MASK_DESCRIPTIONS = MappingProxyType(
    {
        _V1_IDENTITY: "lower_center_fill_post_quantization_global_mean",
        _V2_IDENTITY: (
            "upper_corner_wedges_fill_post_quantization_global_mean"
        ),
    }
)


@dataclass(frozen=True, slots=True)
class DoorPolicyArchitecture:
    hidden_size: int
    num_layers: int


def door_policy_contract_report(
    *,
    hidden_size: int,
    num_layers: int,
) -> dict[str, Any]:
    payload = _payload(
        hidden_size=hidden_size,
        num_layers=num_layers,
        identity=_V2_IDENTITY,
    )
    return payload | {"sha256": _sha256(payload)}


def verify_door_policy_contract(
    report: Mapping[str, Any],
    *,
    hidden_size: int,
    num_layers: int,
) -> None:
    architecture = door_policy_architecture_from_report(report)
    if architecture != DoorPolicyArchitecture(hidden_size, num_layers):
        raise ValueError("fixed-door policy contract does not match runtime")


def door_policy_architecture_from_report(
    report: Mapping[str, Any],
) -> DoorPolicyArchitecture:
    """Validate an approved payload and decode its recurrent architecture."""
    payload = {key: value for key, value in report.items() if key != "sha256"}
    if report.get("sha256") != _sha256(payload):
        raise ValueError("fixed-door policy contract SHA-256 does not match")
    contract_id = payload.get("contract_id")
    schema_version = payload.get("schema_version")
    if not isinstance(contract_id, str) or type(schema_version) is not int:
        raise ValueError("fixed-door policy contract is not approved")
    identity = (contract_id, schema_version)
    if identity not in _APPROVED_SELF_MASK_DESCRIPTIONS:
        raise ValueError("fixed-door policy contract is not approved")
    recurrence = payload.get("recurrence")
    if not isinstance(recurrence, Mapping):
        raise ValueError("fixed-door policy contract has no recurrence")
    try:
        architecture = DoorPolicyArchitecture(
            hidden_size=int(recurrence["hidden_size"]),
            num_layers=int(recurrence["num_layers"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid fixed-door policy architecture") from exc
    expected = _payload(
        hidden_size=architecture.hidden_size,
        num_layers=architecture.num_layers,
        identity=identity,
    )
    if payload != expected:
        raise ValueError("fixed-door policy contract is not an approved payload")
    return architecture


def _payload(
    *,
    hidden_size: int,
    num_layers: int,
    identity: tuple[str, int],
) -> dict[str, Any]:
    if hidden_size <= 0 or num_layers != 1:
        raise ValueError("fixed-door policy requires one positive-size MinGRU layer")
    self_mask = _APPROVED_SELF_MASK_DESCRIPTIONS[identity]
    current_end = DOOR_PIXELS
    delta_end = 2 * DOOR_PIXELS
    motion_end = 3 * DOOR_PIXELS
    sensor_end = motion_end + DOOR_SENSOR_DIM
    phase_end = sensor_end + DOOR_PHASE_DIM
    evidence_end = phase_end + DOOR_EVIDENCE_DIM
    return {
        "contract_id": identity[0],
        "schema_version": identity[1],
        "observation": {
            "total_floats": DOOR_OBS_DIM,
            "deployable_floats": DOOR_POLICY_OBS_DIM,
            "privileged_floats": DOOR_PRIVILEGED_DIM,
            "segments": {
                "current_gray4": [0, current_end],
                "signed_delta": [current_end, delta_end],
                "motion": [delta_end, motion_end],
                "sensors": [motion_end, sensor_end],
                "previous_action": [motion_end + 15, motion_end + 17],
                "phase": [sensor_end, phase_end],
                "detector_evidence": [phase_end, evidence_end],
                "privileged_teacher": [evidence_end, DOOR_OBS_DIM],
            },
            "frame": {
                "width": DOOR_WIDTH,
                "height": DOOR_HEIGHT,
                "host_resize": "PIL_bilinear_grayscale",
                "quantization": "round_to_nearest_multiple_of_17_uint8",
                "self_mask": self_mask,
                "current_scale": "uint8_div_255",
                "delta": "signed_float32_current_minus_previous_div_255",
                "motion": "abs_delta_greater_equal_0.08",
                "reset": "zero_delta_and_motion",
            },
            "sensor_order": [
                "body_vx_div_1",
                "body_vy_div_1",
                "body_vz_div_0.5",
                "body_rate_x_rad_s_div_6",
                "body_rate_y_rad_s_div_6",
                "body_rate_z_rad_s_div_4",
                "body_up_x",
                "body_up_y",
                "body_up_z",
                "altitude_fraction_room_height",
                "origin_forward_displacement_div_4",
                "origin_left_displacement_div_4",
                "origin_vertical_displacement_div_2",
                "sin_relative_yaw",
                "cos_relative_yaw",
                "executed_previous_forward_normalized",
                "executed_previous_yaw_normalized_by_policy_scale",
            ],
            "phase_order": ["search", "track", "approach", "recover"],
            "evidence_order": [
                "confidence",
                "center_x_minus_half_times_2",
                "center_y_minus_half_times_2",
                "sqrt_box_area",
                "normalized_detection_age",
            ],
            "privileged_order": [
                "teacher_forward",
                "teacher_yaw",
                "true_visible",
                "true_center_x",
                "true_center_y",
                "true_scale",
            ],
            "privileged_visible_to_actor": False,
        },
        "recurrence": {
            "kind": "MinGRU",
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "terminal_reset": "zero_after_terminal_step",
            "host_reset": "explicit_between_missions",
        },
        "action_output_order": ["forward", "yaw"],
    }


def _sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
