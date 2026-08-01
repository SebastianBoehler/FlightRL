from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from flightrl.artifact_paths import require_distinct_artifact_paths
from flightrl.evidence_scope import file_identity, require_file_identity
from flightrl.puffer4_edge_checkpoint import (
    EDGE_CHECKPOINT_SCHEMA,
    EdgeCheckpointMetadata,
    load_edge_checkpoint,
)
from flightrl.puffer4_edge_evaluation_evidence import (
    require_edge_evaluation_evidence,
)
from flightrl.puffer4_edge_evaluation_gate import EDGE_EVALUATION_PROFILES
from flightrl.puffer4_edge_schema import EDGE_POLICY_CONTRACT_ID
from flightrl.puffer4_edge_replay import require_edge_passive_replay


EDGE_SHADOW_BUNDLE_SCHEMA = "flightrl.edge_v3.offline_passive_shadow_bundle.v1"
_FIELDS = frozenset(
    {
        "schema",
        "mode",
        "policy_contract_id",
        "policy_contract_sha256",
        "checkpoint_schema",
        "hidden_size",
        "trained_target_ids",
        "checkpoint_identity",
        "evaluation_identity",
        "replay_identity",
        "authority",
        "deployment_authority",
        "hardware_approved",
        "controls_drone",
    }
)


@dataclass(frozen=True, slots=True)
class EdgeShadowBundleMetadata:
    checkpoint: dict[str, str]
    evaluation: dict[str, str]
    replay: dict[str, str]
    hidden_size: int
    trained_target_ids: tuple[int, ...]
    policy_contract_sha256: str


def build_edge_shadow_bundle(
    *,
    checkpoint: str | Path,
    evaluation_report: str | Path,
    replay: str | Path,
) -> dict[str, Any]:
    identities = _capture_input_identities(checkpoint, evaluation_report, replay)
    checkpoint_identity = identities["checkpoint_identity"]
    checkpoint_path = Path(checkpoint_identity["path"])
    actor, metadata = load_edge_checkpoint(checkpoint_path)
    _require_inputs_unchanged(identities, "while loading")
    _require_evaluation(
        identities["evaluation_identity"]["path"],
        checkpoint_identity,
        metadata,
    )
    _require_inputs_unchanged(identities, "during evaluation validation")
    _require_replay(
        identities["replay_identity"]["path"],
        checkpoint_identity,
        actor,
        metadata,
    )
    _require_inputs_unchanged(identities, "during replay validation")
    bundle: dict[str, Any] = {
        "schema": EDGE_SHADOW_BUNDLE_SCHEMA,
        "mode": "offline_passive_shadow",
        "policy_contract_id": EDGE_POLICY_CONTRACT_ID,
        "policy_contract_sha256": metadata.policy_contract_sha256,
        "checkpoint_schema": EDGE_CHECKPOINT_SCHEMA,
        "hidden_size": metadata.hidden_size,
        "trained_target_ids": list(metadata.trained_target_ids),
        **identities,
        "authority": "none",
        "deployment_authority": False,
        "hardware_approved": False,
        "controls_drone": False,
    }
    _require_bundle_envelope(bundle)
    _require_inputs_unchanged(identities, "before bundle return")
    return bundle


def require_edge_shadow_bundle(bundle: object) -> EdgeShadowBundleMetadata:
    _require_bundle_envelope(bundle)
    checkpoint_identity = _identity(bundle.get("checkpoint_identity"), "checkpoint")
    evaluation_identity = _identity(bundle.get("evaluation_identity"), "evaluation")
    replay_identity = _identity(bundle.get("replay_identity"), "replay")
    identities = {
        "checkpoint_identity": checkpoint_identity,
        "evaluation_identity": evaluation_identity,
        "replay_identity": replay_identity,
    }
    actor, checkpoint = load_edge_checkpoint(checkpoint_identity["path"])
    _require_inputs_unchanged(identities, "while loading")
    _require_checkpoint_match(bundle, checkpoint)
    _require_evaluation(
        evaluation_identity["path"],
        checkpoint_identity,
        checkpoint,
    )
    _require_inputs_unchanged(identities, "during evaluation validation")
    _require_replay(
        replay_identity["path"],
        checkpoint_identity,
        actor,
        checkpoint,
    )
    _require_inputs_unchanged(identities, "before bundle return")
    return EdgeShadowBundleMetadata(
        checkpoint=checkpoint_identity,
        evaluation=evaluation_identity,
        replay=replay_identity,
        hidden_size=checkpoint.hidden_size,
        trained_target_ids=checkpoint.trained_target_ids,
        policy_contract_sha256=checkpoint.policy_contract_sha256,
    )


def _require_bundle_envelope(bundle: object) -> None:
    if not isinstance(bundle, Mapping):
        raise TypeError("offline passive-shadow bundle must be a mapping")
    if set(bundle) != _FIELDS:
        raise ValueError("offline passive-shadow bundle fields are incompatible")
    if bundle.get("schema") != EDGE_SHADOW_BUNDLE_SCHEMA:
        raise ValueError("offline passive-shadow bundle schema is incompatible")
    if bundle.get("mode") != "offline_passive_shadow":
        raise ValueError("edge-v3 shadow bundle mode must remain offline and passive")
    if bundle.get("policy_contract_id") != EDGE_POLICY_CONTRACT_ID:
        raise ValueError("edge-v3 shadow policy contract ID is incompatible")
    if bundle.get("checkpoint_schema") != EDGE_CHECKPOINT_SCHEMA:
        raise ValueError("edge-v3 shadow checkpoint schema is incompatible")
    _require_non_authoritative(bundle)


def write_edge_shadow_bundle(bundle: object, path: str | Path) -> None:
    _require_bundle_envelope(bundle)
    identities = {
        field: _identity(bundle.get(field), field.removesuffix("_identity"))
        for field in (
            "checkpoint_identity",
            "evaluation_identity",
            "replay_identity",
        )
    }
    output = Path(path)
    require_distinct_artifact_paths(
        output=output,
        **{
            field.removesuffix("_identity"): identity["path"]
            for field, identity in identities.items()
        },
    )
    _require_inputs_unchanged(identities, "before bundle write")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_edge_shadow_bundle(path: str | Path) -> EdgeShadowBundleMetadata:
    payload = json.loads(Path(path).read_text())
    return require_edge_shadow_bundle(payload)


def _identity(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"offline shadow {label} identity is invalid")
    path = value.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"offline shadow {label} path is invalid")
    return require_file_identity(value, path, label=f"offline shadow {label}")


def _require_checkpoint_match(
    bundle: Mapping[str, object],
    checkpoint: EdgeCheckpointMetadata,
) -> None:
    if bundle.get("hidden_size") != checkpoint.hidden_size:
        raise ValueError("offline shadow hidden size does not match its checkpoint")
    if bundle.get("trained_target_ids") != list(checkpoint.trained_target_ids):
        raise ValueError("offline shadow target IDs do not match its checkpoint")
    if bundle.get("policy_contract_sha256") != checkpoint.policy_contract_sha256:
        raise ValueError("offline shadow contract SHA-256 does not match its checkpoint")


def _require_non_authoritative(bundle: Mapping[str, object]) -> None:
    if (
        bundle.get("authority") != "none"
        or bundle.get("deployment_authority") is not False
        or bundle.get("hardware_approved") is not False
        or bundle.get("controls_drone") is not False
    ):
        raise ValueError(
            "offline passive-shadow bundles can never carry hardware authority or control"
        )


def _require_evaluation(
    path: str | Path,
    checkpoint_identity: Mapping[str, str],
    checkpoint: EdgeCheckpointMetadata,
) -> None:
    require_edge_evaluation_evidence(
        path,
        checkpoint_identity=checkpoint_identity,
        trained_target_ids=checkpoint.trained_target_ids,
        hidden_size=checkpoint.hidden_size,
        policy_contract_sha256=checkpoint.policy_contract_sha256,
        native_build_fingerprint=checkpoint.native_build_fingerprint,
    )


def _require_replay(
    path: str | Path,
    checkpoint_identity: Mapping[str, str],
    actor,
    checkpoint: EdgeCheckpointMetadata,
) -> None:
    checkpoint_path = Path(checkpoint_identity["path"])
    replay = require_edge_passive_replay(
        path,
        checkpoint_context=(checkpoint_path, actor, checkpoint),
    )
    header = replay["header"]
    if (
        header.get("checkpoint_identity") != checkpoint_identity
        or header.get("policy_contract_sha256") != checkpoint.policy_contract_sha256
        or header.get("trained_target_ids") != list(checkpoint.trained_target_ids)
    ):
        raise ValueError("offline shadow replay does not match its checkpoint")
    dataset = replay["dataset_metadata"]
    evaluation_seeds = {profile[1] for profile in EDGE_EVALUATION_PROFILES}
    evaluation_appearance_seeds = {profile[2] for profile in EDGE_EVALUATION_PROFILES}
    if (
        dataset["base_seed"] in evaluation_seeds
        or dataset["appearance_seed"] in evaluation_appearance_seeds
    ):
        raise ValueError(
            "offline shadow replay seeds overlap closed-loop evaluation profiles"
        )
    if (
        dataset["native_build_fingerprint"]
        != checkpoint.native_build_fingerprint
    ):
        raise ValueError("offline shadow replay does not match the training native build")


def _capture_input_identities(checkpoint, evaluation_report, replay) -> dict:
    return {
        "checkpoint_identity": file_identity(checkpoint),
        "evaluation_identity": file_identity(evaluation_report),
        "replay_identity": file_identity(replay),
    }


def _require_inputs_unchanged(identities: Mapping[str, dict], stage: str) -> None:
    for field, identity in identities.items():
        label = field.removesuffix("_identity")
        try:
            current = file_identity(identity["path"])
        except OSError as exc:
            raise RuntimeError(f"offline shadow {label} became unavailable {stage}") from exc
        if current != identity:
            raise RuntimeError(f"offline shadow {label} changed {stage}")
