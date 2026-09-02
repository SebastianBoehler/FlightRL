from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, radians
from typing import Any, Mapping, MutableMapping

from flightrl.artifact_identity import bind_payload, require_bound_payload, sha256_payload
from flightrl.puffer4_edge_schema import ACTION_SPECS


ACTION_ORDER = ("forward", "yaw")
PRIVILEGED_TAIL_ORDER = (
    "teacher_forward",
    "teacher_yaw",
    "visible",
    "center_x",
    "center_y",
    "scale",
)
YAW_POSITIVE = "left"
PREVIOUS_ACTION_FEEDBACK = "simulator_applied_normalized_teacher_action"


@dataclass(frozen=True, slots=True)
class DoorTeacherActionContract:
    contract_id: str
    schema_version: int
    max_forward_speed_m_s: float
    max_yawrate_deg_s: float
    physics_max_yawrate_rad_s: float
    native_yaw_mapping: str

    def __post_init__(self) -> None:
        if not isinstance(self.contract_id, str) or not self.contract_id:
            raise ValueError("fixed-door action contract ID cannot be empty")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported fixed-door action contract schema")
        if (
            isinstance(self.max_forward_speed_m_s, bool)
            or not isfinite(self.max_forward_speed_m_s)
            or self.max_forward_speed_m_s <= 0.0
        ):
            raise ValueError("maximum forward speed must be positive")
        if (
            isinstance(self.max_yawrate_deg_s, bool)
            or not isfinite(self.max_yawrate_deg_s)
            or self.max_yawrate_deg_s <= 0.0
        ):
            raise ValueError("maximum yaw rate must be positive")
        if (
            isinstance(self.physics_max_yawrate_rad_s, bool)
            or not isfinite(self.physics_max_yawrate_rad_s)
            or self.physics_max_yawrate_rad_s <= 0.0
        ):
            raise ValueError("physics yaw ceiling must be positive")
        if self.native_yaw_mapping != "declared_policy_rate":
            raise ValueError(
                "fixed-door action contract must use the declared policy rate"
            )
        if self.native_yaw_action_scale > 1.0 + 1.0e-7:
            raise ValueError("declared yaw rate exceeds physics yaw ceiling")

    @property
    def native_yaw_action_scale(self) -> float:
        """Map normalized policy yaw to the normalized physics setpoint."""
        return radians(self.max_yawrate_deg_s) / self.physics_max_yawrate_rad_s

    def payload(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "schema_version": self.schema_version,
            "action_order": list(ACTION_ORDER),
            "privileged_tail_order": list(PRIVILEGED_TAIL_ORDER),
            "max_forward_speed_m_s": self.max_forward_speed_m_s,
            "max_yawrate_deg_s": self.max_yawrate_deg_s,
            "physics_max_yawrate_rad_s": self.physics_max_yawrate_rad_s,
            "yaw_positive": YAW_POSITIVE,
            "previous_action_feedback": PREVIOUS_ACTION_FEEDBACK,
            "native_yaw_mapping": self.native_yaw_mapping,
        }

    def env_values(self) -> dict[str, float]:
        return {
            "max_horizontal_speed_m_s": self.max_forward_speed_m_s,
            "max_yawrate_deg_s": self.max_yawrate_deg_s,
            "max_rate_yaw": self.physics_max_yawrate_rad_s,
        }

    def apply_to_env(self, env: MutableMapping[str, Any]) -> None:
        env.update(self.env_values())

    def verify_env(self, env: Mapping[str, Any]) -> None:
        for key, expected in self.env_values().items():
            if key not in env or not isclose(
                float(env[key]),
                expected,
                rel_tol=1.0e-7,
                abs_tol=1.0e-7,
            ):
                raise ValueError(
                    f"fixed-door action contract mismatch for {key}: "
                    f"expected {expected}, got {env.get(key)!r}"
                )

    def sha256(self) -> str:
        return sha256_payload(self.payload())

    def to_report(self) -> dict[str, Any]:
        return bind_payload(self.payload())

    @classmethod
    def from_report(cls, report: Mapping[str, Any]) -> DoorTeacherActionContract:
        payload = require_bound_payload(
            report,
            label="fixed-door action contract",
        )
        if tuple(payload.pop("action_order", ())) != ACTION_ORDER:
            raise ValueError("fixed-door action order does not match")
        if (
            tuple(payload.pop("privileged_tail_order", ()))
            != PRIVILEGED_TAIL_ORDER
        ):
            raise ValueError("fixed-door privileged tail order does not match")
        if payload.pop("yaw_positive", None) != YAW_POSITIVE:
            raise ValueError("fixed-door yaw sign does not match")
        if (
            payload.pop("previous_action_feedback", None)
            != PREVIOUS_ACTION_FEEDBACK
        ):
            raise ValueError("fixed-door previous-action feedback does not match")
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ValueError("invalid fixed-door action contract fields") from exc
_EDGE_ACTION_SCALE = {
    name: scale for name, _unit, scale, _frame in ACTION_SPECS
}


PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT = DoorTeacherActionContract(
    contract_id="fixed-door-privileged-teacher-edge-v3-action-envelope",
    schema_version=1,
    max_forward_speed_m_s=_EDGE_ACTION_SCALE["vx"],
    max_yawrate_deg_s=_EDGE_ACTION_SCALE["yaw_rate"],
    physics_max_yawrate_rad_s=4.0,
    native_yaw_mapping="declared_policy_rate",
)

def door_teacher_action_contract_from_report(
    report: Mapping[str, Any],
) -> DoorTeacherActionContract:
    """Decode the single current privileged-teacher action contract."""
    decoded = DoorTeacherActionContract.from_report(report)
    if decoded != PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT:
        raise ValueError(
            "fixed-door teacher action contract is internally valid but not current"
        )
    return PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT
