from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from flightrl.evidence_scope import file_identity, require_file_identity
from flightrl.puffer4_edge_contract import (
    edge_policy_contract_report,
    validate_edge_target_id,
)
from flightrl.puffer4_edge_evidence import require_edge_training_evidence
from flightrl.puffer4_edge_native_build import (
    canonical_edge_native_build_fingerprint,
    require_matching_edge_native_build_fingerprints,
)
from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_schema import EDGE_POLICY_CONTRACT_ID
from flightrl.puffer4_edge_sequence import load_edge_sequence_dataset


EDGE_CHECKPOINT_SCHEMA = "flightrl.edge_v3.checkpoint.v3"
EDGE_CHECKPOINT_FORMAT = "torch_edge_navigation_actor_state_dict"
_FIELDS = frozenset(
    {
        "checkpoint_schema",
        "checkpoint_format",
        "policy_contract_id",
        "policy_contract_sha256",
        "hidden_size",
        "trained_target_ids",
        "dataset_identity",
        "training_identity",
        "native_build_fingerprint",
        "state_dict",
        "authority",
        "deployment_authority",
        "hardware_approved",
        "controls_drone",
    }
)


@dataclass(frozen=True, slots=True)
class EdgeCheckpointMetadata:
    hidden_size: int
    trained_target_ids: tuple[int, ...]
    policy_contract_sha256: str
    dataset_identity: dict[str, str]
    training_identity: dict[str, str]
    native_build_fingerprint: dict


def build_edge_checkpoint_payload(
    actor: EdgeNavigationActor,
    *,
    trained_target_ids: Sequence[int],
    dataset: str | Path,
    training_report: str | Path,
) -> dict[str, Any]:
    if type(actor) is not EdgeNavigationActor:
        raise TypeError("edge checkpoint actor must be an EdgeNavigationActor")
    hidden_size = actor.hidden_size
    dataset_identity = file_identity(dataset)
    selection = load_edge_sequence_dataset(dataset_identity["path"])
    if file_identity(dataset) != dataset_identity:
        raise RuntimeError("edge checkpoint dataset changed while loading")
    payload: dict[str, Any] = {
        "checkpoint_schema": EDGE_CHECKPOINT_SCHEMA,
        "checkpoint_format": EDGE_CHECKPOINT_FORMAT,
        "policy_contract_id": EDGE_POLICY_CONTRACT_ID,
        "policy_contract_sha256": edge_policy_contract_report(hidden_size=hidden_size)[
            "sha256"
        ],
        "hidden_size": hidden_size,
        "trained_target_ids": list(_target_ids(trained_target_ids)),
        "dataset_identity": dataset_identity,
        "training_identity": file_identity(training_report),
        "native_build_fingerprint": canonical_edge_native_build_fingerprint(
            selection.metadata["native_build_fingerprint"]
        ),
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in actor.state_dict().items()
        },
        "authority": "none",
        "deployment_authority": False,
        "hardware_approved": False,
        "controls_drone": False,
    }
    require_edge_checkpoint(payload)
    return payload


def require_edge_checkpoint(checkpoint: object) -> EdgeCheckpointMetadata:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("edge-v3 checkpoint must be a mapping envelope")
    if set(checkpoint) != _FIELDS:
        raise ValueError(
            "edge-v3 checkpoint fields are missing or incompatible; raw, partial, and legacy checkpoints are rejected"
        )
    if checkpoint.get("checkpoint_schema") != EDGE_CHECKPOINT_SCHEMA:
        raise ValueError("edge-v3 checkpoint schema is incompatible")
    if checkpoint.get("checkpoint_format") != EDGE_CHECKPOINT_FORMAT:
        raise ValueError("edge-v3 checkpoint format is incompatible")
    if checkpoint.get("policy_contract_id") != EDGE_POLICY_CONTRACT_ID:
        raise ValueError("edge-v3 policy contract ID is incompatible")
    _require_non_authoritative(checkpoint)

    hidden_size = _hidden_size(checkpoint.get("hidden_size"))
    expected_contract = edge_policy_contract_report(hidden_size=hidden_size)
    contract_sha = checkpoint.get("policy_contract_sha256")
    if contract_sha != expected_contract["sha256"]:
        raise ValueError("edge-v3 checkpoint policy contract SHA-256 is incompatible")
    target_ids = _target_ids(checkpoint.get("trained_target_ids"))
    identities = tuple(
        _require_identity(checkpoint.get(field), field)
        for field in (
            "dataset_identity",
            "training_identity",
        )
    )
    _require_exact_state_dict(checkpoint.get("state_dict"), hidden_size)
    training_fingerprint = require_edge_training_evidence(
        identities[1]["path"],
        selection_identity=identities[0],
        hidden_size=hidden_size,
        trained_target_ids=target_ids,
        policy_contract_sha256=contract_sha,
        actor_state_dict=checkpoint["state_dict"],
    )
    native_fingerprint = require_matching_edge_native_build_fingerprints(
        checkpoint["native_build_fingerprint"], training_fingerprint
    )
    return EdgeCheckpointMetadata(
        hidden_size=hidden_size,
        trained_target_ids=target_ids,
        policy_contract_sha256=contract_sha,
        dataset_identity=identities[0],
        training_identity=identities[1],
        native_build_fingerprint=native_fingerprint,
    )


def save_edge_checkpoint(checkpoint: object, path: str | Path) -> None:
    require_edge_checkpoint(checkpoint)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(checkpoint), output)


def load_edge_checkpoint(
    path: str | Path,
) -> tuple[EdgeNavigationActor, EdgeCheckpointMetadata]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    metadata = require_edge_checkpoint(checkpoint)
    actor = EdgeNavigationActor(hidden_size=metadata.hidden_size)
    actor.load_state_dict(dict(checkpoint["state_dict"]), strict=True)
    actor.eval()
    return actor, metadata


def _hidden_size(value: object) -> int:
    if type(value) is not int or not 32 <= value <= 64:
        raise ValueError(
            "edge-v3 checkpoint hidden_size must be an integer in [32, 64]"
        )
    return value


def _target_ids(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("edge-v3 trained target IDs must be a sequence")
    target_ids = tuple(value)
    if not target_ids or any(type(target_id) is not int for target_id in target_ids):
        raise ValueError("edge-v3 trained target IDs must be nonempty exact integers")
    if tuple(sorted(set(target_ids))) != target_ids:
        raise ValueError("edge-v3 trained target IDs must be sorted and unique")
    for target_id in target_ids:
        validate_edge_target_id(target_id)
    return target_ids


def _require_identity(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"edge-v3 checkpoint {label} is invalid")
    path = value.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"edge-v3 checkpoint {label} path is invalid")
    return require_file_identity(value, path, label=f"edge-v3 {label}")


def _require_exact_state_dict(state_dict: object, hidden_size: int) -> None:
    if not isinstance(state_dict, Mapping):
        raise ValueError("edge-v3 checkpoint state_dict must be a mapping")
    expected = EdgeNavigationActor(hidden_size=hidden_size).state_dict()
    if set(state_dict) != set(expected):
        raise ValueError(
            "edge-v3 checkpoint state_dict keys do not exactly match the actor"
        )
    for key, reference in expected.items():
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"edge-v3 checkpoint tensor {key!r} is invalid")
        if value.dtype != reference.dtype or value.shape != reference.shape:
            raise ValueError(
                f"edge-v3 checkpoint tensor {key!r} shape or dtype is incompatible"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"edge-v3 checkpoint tensor {key!r} is nonfinite")


def _require_non_authoritative(checkpoint: Mapping[str, object]) -> None:
    if (
        checkpoint.get("authority") != "none"
        or checkpoint.get("deployment_authority") is not False
        or checkpoint.get("hardware_approved") is not False
        or checkpoint.get("controls_drone") is not False
    ):
        raise ValueError(
            "edge-v3 desktop checkpoint must be explicitly non-authoritative"
        )
