from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isclose, isfinite
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping


@dataclass(frozen=True, slots=True)
class DoorEvidenceAgeContract:
    contract_id: str
    schema_version: int
    control_dt_s: float
    maximum_evidence_age_s: float
    simulator_age_origin: str
    host_age_origin: str
    normalized_encoding: str
    stale_boundary: str

    def __post_init__(self) -> None:
        if not self.contract_id:
            raise ValueError("fixed-door evidence-age contract ID cannot be empty")
        if self.schema_version != 1:
            raise ValueError("unsupported fixed-door evidence-age schema")
        if not isfinite(self.control_dt_s) or self.control_dt_s <= 0.0:
            raise ValueError("fixed-door control period must be finite and positive")
        if (
            not isfinite(self.maximum_evidence_age_s)
            or self.maximum_evidence_age_s <= 0.0
        ):
            raise ValueError(
                "fixed-door maximum evidence age must be finite and positive"
            )
        if self.simulator_age_origin != "latest_detector_update_attempt":
            raise ValueError("unsupported simulator evidence-age origin")
        if self.host_age_origin != "latest_detector_source_frame":
            raise ValueError("unsupported host evidence-age origin")
        if self.normalized_encoding != "clip(elapsed_s/maximum_age_s,0,1)":
            raise ValueError("unsupported fixed-door evidence-age encoding")
        if self.stale_boundary != "normalized_age_greater_equal_1":
            raise ValueError("unsupported fixed-door stale boundary")

    def payload(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "schema_version": self.schema_version,
            "control_dt_s": self.control_dt_s,
            "maximum_evidence_age_s": self.maximum_evidence_age_s,
            "simulator_age_origin": self.simulator_age_origin,
            "host_age_origin": self.host_age_origin,
            "normalized_encoding": self.normalized_encoding,
            "stale_boundary": self.stale_boundary,
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

    def env_values(self) -> dict[str, float]:
        return {
            "control_dt": self.control_dt_s,
            "maximum_evidence_age_s": self.maximum_evidence_age_s,
        }

    def apply_to_env(self, env: MutableMapping[str, Any]) -> None:
        env.update(self.env_values())

    def verify_env(self, env: Mapping[str, Any]) -> None:
        for key, expected in self.env_values().items():
            value = env.get(key)
            try:
                matches = isfinite(float(value)) and isclose(
                    float(value),
                    expected,
                    rel_tol=1.0e-7,
                    abs_tol=1.0e-9,
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise ValueError(
                    f"fixed-door evidence-age contract mismatch for {key}: "
                    f"expected {expected}, got {value!r}"
                )

    @classmethod
    def from_report(
        cls,
        report: Mapping[str, Any],
    ) -> DoorEvidenceAgeContract:
        payload = {key: value for key, value in report.items() if key != "sha256"}
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if report.get("sha256") != hashlib.sha256(encoded).hexdigest():
            raise ValueError(
                "fixed-door evidence-age contract SHA-256 does not match"
            )
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ValueError(
                "invalid fixed-door evidence-age contract fields"
            ) from exc


FIXED_DOOR_EVIDENCE_AGE_CONTRACT = DoorEvidenceAgeContract(
    contract_id="fixed-door-evidence-age-runtime-v1",
    schema_version=1,
    control_dt_s=1.0 / 65.0,
    maximum_evidence_age_s=1.0,
    simulator_age_origin="latest_detector_update_attempt",
    host_age_origin="latest_detector_source_frame",
    normalized_encoding="clip(elapsed_s/maximum_age_s,0,1)",
    stale_boundary="normalized_age_greater_equal_1",
)

APPROVED_DOOR_EVIDENCE_AGE_CONTRACTS = MappingProxyType(
    {
        FIXED_DOOR_EVIDENCE_AGE_CONTRACT.contract_id: (
            FIXED_DOOR_EVIDENCE_AGE_CONTRACT
        )
    }
)


def approved_door_evidence_age_contract_from_report(
    report: Mapping[str, Any],
) -> DoorEvidenceAgeContract:
    decoded = DoorEvidenceAgeContract.from_report(report)
    approved = APPROVED_DOOR_EVIDENCE_AGE_CONTRACTS.get(decoded.contract_id)
    if approved is None or decoded != approved:
        raise ValueError(
            "fixed-door evidence-age contract is internally valid but not approved"
        )
    return approved
