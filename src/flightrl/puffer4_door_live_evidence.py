from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Literal, Mapping

from flightrl.puffer4_door_bundle import (
    FixedDoorCheckpointBundle,
    load_fixed_door_checkpoint_bundle,
)
from flightrl.puffer4_door_contract import (
    CORRECTED_DOOR_ACTION_CONTRACT,
    FIXED_DOOR_LIVE_SAFETY_CONTRACT,
    LEGACY_V59_ACTION_CONTRACT,
)
from flightrl.puffer4_door_evidence_age_contract import (
    approved_door_evidence_age_contract_from_report,
)
from flightrl.puffer4_door_native_evidence import (
    validate_native_build_fingerprint,
)
from flightrl.puffer4_door_stream_contract import verify_door_stream_contract


PROMOTION_EVALUATION_SCHEMA = "flightrl.fixed_door.promotion.v3"
V59_CHECKPOINT_SHA256 = (
    "f676d12b9d37c27f4cc62f99beceec8f30e74c88be8564cb242c23755e202cce"
)
V59_REEVALUATION_SHA256 = (
    "b919e4f9951ad28904ce6cc7ee9b7a0f7b76ee70fba387e673bcda27a9bdcbbc"
)
LiveEvidenceKind = Literal["promotion_v3", "grandfathered_v59"]


@dataclass(frozen=True, slots=True)
class FixedDoorLiveEvidence:
    kind: LiveEvidenceKind
    bundle: FixedDoorCheckpointBundle


def validate_fixed_door_live_evidence(
    checkpoint: str | Path,
    report_path: str | Path,
) -> FixedDoorLiveEvidence:
    """Validate immutable simulation evidence before any real-world gate."""
    bundle = load_fixed_door_checkpoint_bundle(checkpoint, report_path)
    if (
        bundle.checkpoint_sha256 == V59_CHECKPOINT_SHA256
        and bundle.report_sha256 == V59_REEVALUATION_SHA256
    ):
        _validate_grandfathered_v59(bundle)
        return FixedDoorLiveEvidence("grandfathered_v59", bundle)
    _validate_promotion_v3(bundle)
    return FixedDoorLiveEvidence("promotion_v3", bundle)


def _validate_grandfathered_v59(bundle: FixedDoorCheckpointBundle) -> None:
    if bundle.report_sha256 != V59_REEVALUATION_SHA256:
        raise ValueError("v59 requires the exact authoritative reevaluation")
    if bundle.action_contract != LEGACY_V59_ACTION_CONTRACT:
        raise ValueError("v59 action contract does not match its grandfathered SHA")


def _validate_promotion_v3(bundle: FixedDoorCheckpointBundle) -> None:
    report = bundle.raw_report
    if report.get("evaluation_schema") != PROMOTION_EVALUATION_SCHEMA:
        raise ValueError("live evidence requires a promotion.v3 evaluation")
    if not isinstance(report.get("trained_identity"), Mapping):
        raise ValueError("promotion.v3 requires a nested trained identity")
    if bundle.lineage_report_path == bundle.report_path:
        raise ValueError("promotion.v3 requires hash-bound nested lineage")
    legacy_v59 = (
        bundle.checkpoint_sha256 == V59_CHECKPOINT_SHA256
        and bundle.action_contract == LEGACY_V59_ACTION_CONTRACT
    )
    if bundle.action_contract != CORRECTED_DOOR_ACTION_CONTRACT and not legacy_v59:
        raise ValueError("only v59 may use the legacy action contract")

    identity = _mapping(report.get("evaluation_identity"), "evaluation identity")
    _validate_evaluation_identity(bundle, identity)
    validate_native_build_fingerprint(
        _mapping(
            identity.get("native_build_fingerprint"),
            "native build fingerprint",
        ),
        bundle.env_name,
    )
    _validate_promotion_runs(report, identity)
    _validate_live_yaw_cap(report, bundle)


def _validate_evaluation_identity(
    bundle: FixedDoorCheckpointBundle,
    identity: Mapping[str, Any],
) -> None:
    if identity.get("action_contract_sha256") != bundle.action_contract.sha256():
        raise ValueError("evaluation action contract identity does not match")
    if identity.get("policy_contract_sha256") != bundle.policy_contract["sha256"]:
        raise ValueError("evaluation policy contract identity does not match")
    environment = _mapping(identity.get("environment"), "evaluation environment")
    if environment.get("name") != bundle.env_name:
        raise ValueError("evaluation environment identity does not match")
    _positive_int(environment.get("agents"), "evaluation agents")
    _positive_int(
        environment.get("steps_per_condition"),
        "evaluation steps per condition",
    )
    seed = environment.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("evaluation seed is invalid")

    stream = _mapping(
        identity.get("procedural_stream_contract"),
        "evaluation procedural stream contract",
    )
    verify_door_stream_contract(stream)
    stream_matches_lineage = (
        bundle.stream_contract is not None
        and dict(stream) == dict(bundle.stream_contract)
    )
    legacy_v59_without_stream = (
        bundle.checkpoint_sha256 == V59_CHECKPOINT_SHA256
        and bundle.stream_contract is None
    )
    if not stream_matches_lineage and not legacy_v59_without_stream:
        raise ValueError("evaluation procedural stream identity does not match")
    evidence_age = _mapping(
        identity.get("evidence_age_runtime_contract"),
        "evaluation evidence-age runtime contract",
    )
    approved_door_evidence_age_contract_from_report(evidence_age)


def _validate_promotion_runs(
    report: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    steps = _mapping(identity.get("environment"), "evaluation environment")[
        "steps_per_condition"
    ]
    _require_complete_run(report.get("full_camera"), "full-camera", steps)
    _require_complete_run(report.get("masked_camera"), "masked-camera", steps)
    recurrence = _mapping(
        report.get("recurrence_reset_ablation"),
        "recurrence-reset ablation",
    )
    _require_complete_run(
        recurrence.get("metrics"),
        "recurrence-reset",
        steps,
    )
    temporal = _mapping(
        report.get("temporal_order_ablation"),
        "temporal-order ablation",
    )
    _require_complete_run(temporal.get("metrics"), "temporal-order", steps)
    promotion = _mapping(report.get("promotion_evidence"), "promotion evidence")
    required = (
        "all_default_runs_complete",
        "all_default_outputs_finite",
        "temporal_order_run_complete",
        "temporal_order_outputs_finite",
    )
    if any(promotion.get(key) is not True for key in required):
        raise ValueError("promotion evidence is incomplete or non-finite")
    gate = _mapping(report.get("simulation_gate"), "simulation gate")
    checks = _mapping(gate.get("checks"), "simulation gate checks")
    if (
        gate.get("passed") is not True
        or not checks
        or any(value is not True for value in checks.values())
        or gate.get("failures") != []
    ):
        raise ValueError("simulation promotion gate has not passed")


def _validate_live_yaw_cap(
    report: Mapping[str, Any],
    bundle: FixedDoorCheckpointBundle,
) -> None:
    challenge = _mapping(
        report.get("live_yaw_cap_challenge"),
        "live yaw-cap challenge",
    )
    live_limit = FIXED_DOOR_LIVE_SAFETY_CONTRACT.max_yawrate_deg_s
    expected_normalized = FIXED_DOOR_LIVE_SAFETY_CONTRACT.normalized_yaw_limit(
        bundle.action_contract
    )
    if (
        challenge.get("label") != "live_yaw_cap_only"
        or not _close(
            challenge.get("policy_max_yawrate_deg_s"),
            bundle.action_contract.max_yawrate_deg_s,
        )
        or not _close(challenge.get("live_max_yawrate_deg_s"), live_limit)
        or not _close(
            challenge.get("normalized_yaw_limit"),
            expected_normalized,
        )
    ):
        raise ValueError("live yaw-cap challenge scale is invalid")
    condition = _mapping(
        challenge.get("condition"),
        "live yaw-cap challenge condition",
    )
    expected_condition = {
        "camera": "full",
        "recurrent_mode": "carried",
        "forward_action": "unchanged",
        "combined_with_other_ablation": False,
    }
    if dict(condition) != expected_condition:
        raise ValueError("live yaw-cap challenge is not a single intervention")
    metrics = _require_complete_run(
        challenge.get("metrics"),
        "live yaw-cap",
        _mapping(
            report.get("evaluation_identity"),
            "evaluation identity",
        )["environment"]["steps_per_condition"],
    )
    yaw_cap = _mapping(metrics.get("yaw_cap"), "live yaw-cap metrics")
    if (
        yaw_cap.get("enabled") is not True
        or not _close(yaw_cap.get("normalized_limit"), expected_normalized)
    ):
        raise ValueError("live yaw-cap metrics do not prove the configured cap")


def _require_complete_run(
    value: object,
    label: str,
    expected_steps: int,
) -> Mapping[str, Any]:
    run = _mapping(value, f"{label} metrics")
    finite = _mapping(run.get("finite_outputs"), f"{label} finite outputs")
    required_finite = (
        "observations",
        "terminals",
        "policy_mean",
        "value",
        "recurrent_state",
        "actions",
        "metrics",
        "passed",
    )
    if (
        run.get("status") != "complete"
        or run.get("requested_steps") != expected_steps
        or run.get("completed_steps") != expected_steps
        or any(finite.get(key) is not True for key in required_finite)
        or finite.get("first_violation") is not None
    ):
        raise ValueError(f"{label} run is incomplete or non-finite")
    _require_finite_numbers(run, label)
    return run


def _require_finite_numbers(value: object, label: str) -> None:
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite_numbers(item, label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_finite_numbers(item, label)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing or invalid")
    return value


def _close(value: object, expected: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
        and isclose(float(value), expected, rel_tol=1.0e-9, abs_tol=1.0e-12)
    )
