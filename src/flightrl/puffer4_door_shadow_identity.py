from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_door_live_evidence import FixedDoorLiveEvidence
from flightrl.puffer4_door_shadow_detector_contract import (
    APPROVED_SHADOW_DETECTOR_MODEL_ID,
    APPROVED_SHADOW_DEVICE,
    APPROVED_SHADOW_HARDWARE_CONFIG,
    APPROVED_SHADOW_PROMPT,
    APPROVED_SHADOW_THRESHOLD,
    approved_shadow_detector_contract,
    approved_shadow_hardware_config_identity,
    require_approved_shadow_runtime,
)


SHADOW_IDENTITY_JSON_FIELD = "shadow_run_identity_json"
SHADOW_IDENTITY_SHA256_FIELD = "shadow_run_identity_sha256"
SHADOW_IDENTITY_SCHEMA = "flightrl.fixed_door.real_shadow.v1"


@dataclass(frozen=True, slots=True)
class EncodedShadowIdentity:
    payload: Mapping[str, Any]
    canonical_json: str
    sha256: str


def build_fixed_door_shadow_identity(
    evidence: FixedDoorLiveEvidence,
    *,
    prompt: str,
    detector_model_id: str,
    threshold: float,
    device: str,
    hardware_config: str | Path,
) -> EncodedShadowIdentity:
    require_approved_shadow_runtime(
        prompt=prompt,
        detector_model_id=detector_model_id,
        threshold=threshold,
        device=device,
        hardware_config=hardware_config,
    )
    bundle = evidence.bundle
    payload = {
        "schema": SHADOW_IDENTITY_SCHEMA,
        "live_evidence_kind": evidence.kind,
        "checkpoint": {
            "path": str(bundle.checkpoint_path),
            "sha256": bundle.checkpoint_sha256,
        },
        "evaluation_report": {
            "path": str(bundle.report_path),
            "sha256": bundle.report_sha256,
        },
        "lineage_report": {
            "path": str(bundle.lineage_report_path),
            "sha256": bundle.lineage_report_sha256,
        },
        "action_contract": {
            "contract_id": bundle.action_contract.contract_id,
            "sha256": bundle.action_contract.sha256(),
        },
        "policy_contract": {
            "contract_id": bundle.policy_contract["contract_id"],
            "sha256": bundle.policy_contract["sha256"],
        },
        "evidence_age_contract": {
            "contract_id": (
                FIXED_DOOR_EVIDENCE_AGE_CONTRACT.contract_id
            ),
            "sha256": FIXED_DOOR_EVIDENCE_AGE_CONTRACT.sha256(),
        },
        "detector_contract": approved_shadow_detector_contract(),
        "inference_device": device,
        "hardware_config": approved_shadow_hardware_config_identity(),
        "monitor_only": True,
        "controls_drone": False,
    }
    return _encode(payload)


def decode_fixed_door_shadow_identity(
    canonical_json: str,
    digest: str,
) -> EncodedShadowIdentity:
    try:
        payload = json.loads(canonical_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("shadow run identity JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("shadow run identity must be a JSON object")
    encoded = _canonical_json(payload)
    if canonical_json != encoded:
        raise ValueError("shadow run identity JSON is not canonical")
    if digest != hashlib.sha256(encoded.encode()).hexdigest():
        raise ValueError("shadow run identity SHA-256 does not match")
    _validate_identity_payload(payload)
    return EncodedShadowIdentity(payload, encoded, digest)


def require_shadow_identity_matches_evidence(
    identity: EncodedShadowIdentity,
    evidence: FixedDoorLiveEvidence,
) -> None:
    expected = build_fixed_door_shadow_identity(
        evidence,
        prompt=APPROVED_SHADOW_PROMPT,
        detector_model_id=APPROVED_SHADOW_DETECTOR_MODEL_ID,
        threshold=APPROVED_SHADOW_THRESHOLD,
        device=APPROVED_SHADOW_DEVICE,
        hardware_config=APPROVED_SHADOW_HARDWARE_CONFIG,
    )
    if identity != expected:
        fields = (
            "checkpoint",
            "evaluation_report",
            "lineage_report",
            "action_contract",
            "policy_contract",
            "evidence_age_contract",
            "detector_contract",
            "inference_device",
            "hardware_config",
            "live_evidence_kind",
        )
        changed = [
            field
            for field in fields
            if identity.payload.get(field) != expected.payload.get(field)
        ]
        raise ValueError(
            "shadow run identity does not match live evidence: "
            + ", ".join(changed or ("canonical identity",))
        )


def _validate_identity_payload(payload: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "live_evidence_kind",
        "checkpoint",
        "evaluation_report",
        "lineage_report",
        "action_contract",
        "policy_contract",
        "evidence_age_contract",
        "detector_contract",
        "inference_device",
        "hardware_config",
        "monitor_only",
        "controls_drone",
    }
    if set(payload) != expected_keys:
        raise ValueError("shadow run identity fields are incomplete")
    if payload.get("schema") != SHADOW_IDENTITY_SCHEMA:
        raise ValueError("shadow run identity schema is not approved")
    if payload.get("live_evidence_kind") not in {
        "grandfathered_v59",
        "promotion_v3",
    }:
        raise ValueError("shadow live evidence kind is not approved")
    if (
        payload.get("monitor_only") is not True
        or payload.get("controls_drone") is not False
    ):
        raise ValueError("shadow run identity must be non-actuating")
    for label in ("checkpoint", "evaluation_report", "lineage_report"):
        _validate_file_identity(payload.get(label), label)
    for label in ("action_contract", "policy_contract"):
        _validate_contract_identity(payload.get(label), label)
    evidence_age = payload.get("evidence_age_contract")
    expected_age = {
        "contract_id": FIXED_DOOR_EVIDENCE_AGE_CONTRACT.contract_id,
        "sha256": FIXED_DOOR_EVIDENCE_AGE_CONTRACT.sha256(),
    }
    if evidence_age != expected_age:
        raise ValueError("shadow evidence-age contract is not approved")
    if payload.get("detector_contract") != approved_shadow_detector_contract():
        raise ValueError("shadow detector contract is not approved")
    if payload.get("inference_device") != APPROVED_SHADOW_DEVICE:
        raise ValueError("shadow inference device is not approved")
    if (
        payload.get("hardware_config")
        != approved_shadow_hardware_config_identity()
    ):
        raise ValueError("shadow hardware config is not approved")


def _validate_file_identity(value: object, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"shadow {label} identity is invalid")
    path = value.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError(f"shadow {label} path is invalid")
    _require_sha256(value.get("sha256"), f"shadow {label}")


def _validate_contract_identity(value: object, label: str) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"contract_id", "sha256"}
        or not isinstance(value.get("contract_id"), str)
        or not value.get("contract_id")
    ):
        raise ValueError(f"shadow {label} identity is invalid")
    _require_sha256(value.get("sha256"), f"shadow {label}")


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} SHA-256 is invalid")


def _encode(payload: Mapping[str, Any]) -> EncodedShadowIdentity:
    canonical = _canonical_json(payload)
    return EncodedShadowIdentity(
        dict(payload),
        canonical,
        hashlib.sha256(canonical.encode()).hexdigest(),
    )


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))

