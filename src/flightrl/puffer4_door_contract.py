from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import degrees, isclose, isfinite, radians
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping


ACTION_ORDER = ("forward", "yaw")
YAW_POSITIVE = "left"
PREVIOUS_ACTION_FEEDBACK = "executed_normalized_by_policy_scale"


@dataclass(frozen=True, slots=True)
class DoorActionContract:
    contract_id: str
    schema_version: int
    max_forward_speed_m_s: float
    max_yawrate_deg_s: float
    physics_max_yawrate_rad_s: float
    native_yaw_mapping: str

    def __post_init__(self) -> None:
        if not self.contract_id:
            raise ValueError("fixed-door action contract ID cannot be empty")
        if self.schema_version != 1:
            raise ValueError("unsupported fixed-door action contract schema")
        if (
            not isfinite(self.max_forward_speed_m_s)
            or self.max_forward_speed_m_s <= 0.0
        ):
            raise ValueError("maximum forward speed must be positive")
        if not isfinite(self.max_yawrate_deg_s) or self.max_yawrate_deg_s <= 0.0:
            raise ValueError("maximum yaw rate must be positive")
        if (
            not isfinite(self.physics_max_yawrate_rad_s)
            or self.physics_max_yawrate_rad_s <= 0.0
        ):
            raise ValueError("physics yaw ceiling must be positive")
        if self.native_yaw_mapping not in {
            "declared_policy_rate",
            "legacy_direct_physics_ceiling",
        }:
            raise ValueError("unknown native yaw mapping")
        if self.native_yaw_action_scale > 1.0 + 1.0e-7:
            raise ValueError("declared yaw rate exceeds physics yaw ceiling")
        if (
            self.native_yaw_mapping == "legacy_direct_physics_ceiling"
            and not isclose(
                self.native_yaw_action_scale,
                1.0,
                rel_tol=1.0e-7,
                abs_tol=1.0e-7,
            )
        ):
            raise ValueError("legacy yaw mapping must equal physics yaw ceiling")

    @property
    def native_yaw_action_scale(self) -> float:
        """Map normalized policy yaw to the normalized physics setpoint."""
        return radians(self.max_yawrate_deg_s) / self.physics_max_yawrate_rad_s

    def payload(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "schema_version": self.schema_version,
            "action_order": list(ACTION_ORDER),
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
        encoded = json.dumps(
            self.payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_report(self) -> dict[str, Any]:
        return self.payload() | {"sha256": self.sha256()}

    @classmethod
    def from_report(cls, report: Mapping[str, Any]) -> DoorActionContract:
        payload = {key: value for key, value in report.items() if key != "sha256"}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        if report.get("sha256") != digest:
            raise ValueError("fixed-door action contract SHA-256 does not match")
        if tuple(payload.pop("action_order", ())) != ACTION_ORDER:
            raise ValueError("fixed-door action order does not match")
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


@dataclass(frozen=True, slots=True)
class DoorLiveSafetyContract:
    contract_id: str
    schema_version: int
    max_yawrate_deg_s: float
    proposal_stale_s: float
    min_height_m: float
    max_height_m: float
    max_duration_s: float
    translation_enabled: bool

    def __post_init__(self) -> None:
        if not self.contract_id:
            raise ValueError("fixed-door live safety contract ID cannot be empty")
        if self.schema_version != 2:
            raise ValueError("unsupported fixed-door live safety schema")
        if not isfinite(self.max_yawrate_deg_s) or self.max_yawrate_deg_s <= 0.0:
            raise ValueError("live yaw limit must be positive")
        if not isfinite(self.proposal_stale_s) or self.proposal_stale_s <= 0.0:
            raise ValueError("proposal stale limit must be positive")
        if (
            not isfinite(self.min_height_m)
            or not isfinite(self.max_height_m)
            or self.min_height_m <= 0.0
            or self.max_height_m < self.min_height_m
        ):
            raise ValueError("live height envelope is invalid")
        if not isfinite(self.max_duration_s) or self.max_duration_s <= 0.0:
            raise ValueError("live duration limit must be positive")
        if self.translation_enabled:
            raise ValueError("fixed-door yaw gate cannot enable translation")

    def payload(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "schema_version": self.schema_version,
            "max_yawrate_deg_s": self.max_yawrate_deg_s,
            "proposal_stale_s": self.proposal_stale_s,
            "min_height_m": self.min_height_m,
            "max_height_m": self.max_height_m,
            "max_duration_s": self.max_duration_s,
            "translation_enabled": self.translation_enabled,
        }

    def sha256(self) -> str:
        encoded = json.dumps(
            self.payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_report(self) -> dict[str, Any]:
        return self.payload() | {"sha256": self.sha256()}

    @classmethod
    def from_report(
        cls,
        report: Mapping[str, Any],
    ) -> DoorLiveSafetyContract:
        payload = {key: value for key, value in report.items() if key != "sha256"}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        if report.get("sha256") != hashlib.sha256(encoded).hexdigest():
            raise ValueError("fixed-door live safety contract SHA-256 does not match")
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ValueError("invalid fixed-door live safety contract fields") from exc

    def readiness_limits(self) -> dict[str, float]:
        return {
            "search_abs_yawrate_deg_s": self.max_yawrate_deg_s,
            "detected_abs_yawrate_deg_s": self.max_yawrate_deg_s,
            "proposal_stale_s": self.proposal_stale_s,
            "min_height_m": self.min_height_m,
            "max_height_m": self.max_height_m,
            "max_duration_s": self.max_duration_s,
        }

    def normalized_yaw_limit(self, action: DoorActionContract) -> float:
        if self.max_yawrate_deg_s > action.max_yawrate_deg_s:
            raise ValueError("live yaw limit exceeds policy yaw scale")
        return self.max_yawrate_deg_s / action.max_yawrate_deg_s

    def require_live_envelope(
        self,
        *,
        height_m: float,
        duration_s: float,
    ) -> None:
        if (
            isinstance(height_m, bool)
            or not isfinite(height_m)
            or not self.min_height_m <= height_m <= self.max_height_m
        ):
            raise ValueError(
                "fixed-door live height must be within "
                f"[{self.min_height_m}, {self.max_height_m}] m"
            )
        if (
            isinstance(duration_s, bool)
            or not isfinite(duration_s)
            or not 0.0 < duration_s <= self.max_duration_s
        ):
            raise ValueError(
                "fixed-door live duration must be within "
                f"(0, {self.max_duration_s}] s"
            )


CORRECTED_DOOR_ACTION_CONTRACT = DoorActionContract(
    contract_id="fixed-door-declared-yaw-v1",
    schema_version=1,
    max_forward_speed_m_s=0.55,
    max_yawrate_deg_s=70.0,
    physics_max_yawrate_rad_s=4.0,
    native_yaw_mapping="declared_policy_rate",
)

LEGACY_V59_ACTION_CONTRACT = DoorActionContract(
    contract_id="fixed-door-v59-legacy-physics-yaw-v1",
    schema_version=1,
    max_forward_speed_m_s=0.55,
    max_yawrate_deg_s=degrees(4.0),
    physics_max_yawrate_rad_s=4.0,
    native_yaw_mapping="legacy_direct_physics_ceiling",
)

FIXED_DOOR_LIVE_SAFETY_CONTRACT = DoorLiveSafetyContract(
    contract_id="fixed-door-yaw-only-live-v2",
    schema_version=2,
    max_yawrate_deg_s=8.0,
    proposal_stale_s=0.75,
    min_height_m=0.20,
    max_height_m=0.80,
    max_duration_s=15.0,
    translation_enabled=False,
)

APPROVED_DOOR_ACTION_CONTRACTS = MappingProxyType(
    {
        contract.contract_id: contract
        for contract in (
            CORRECTED_DOOR_ACTION_CONTRACT,
            LEGACY_V59_ACTION_CONTRACT,
        )
    }
)


def approved_door_action_contract_from_report(
    report: Mapping[str, Any],
) -> DoorActionContract:
    """Decode only a reviewed, immutable fixed-door action contract."""
    decoded = DoorActionContract.from_report(report)
    approved = APPROVED_DOOR_ACTION_CONTRACTS.get(decoded.contract_id)
    if approved is None or decoded != approved:
        raise ValueError(
            "fixed-door action contract is internally valid but not approved"
        )
    return approved
