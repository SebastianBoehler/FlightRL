from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from flightrl.puffer4_door_contract import (
    DoorActionContract,
    approved_door_action_contract_from_report,
)
from flightrl.puffer4_door_policy_contract import (
    DoorPolicyArchitecture,
    door_policy_architecture_from_report,
)
from flightrl.puffer4_door_stream_contract import verify_door_stream_contract


FIXED_DOOR_ENV_NAME = "flightrl_fixed_door_d1"


@dataclass(frozen=True, slots=True)
class FixedDoorCheckpointBundle:
    checkpoint_path: Path
    checkpoint_sha256: str
    report_path: Path
    report_sha256: str
    action_contract: DoorActionContract
    policy_contract: Mapping[str, Any]
    architecture: DoorPolicyArchitecture
    env_name: str
    train_seed: int
    stream_contract: Mapping[str, Any] | None
    lineage_report_path: Path
    lineage_report_sha256: str
    raw_report: Mapping[str, Any]

    def trained_identity(self) -> dict[str, Any]:
        identity = {
            "checkpoint": {
                "path": str(self.checkpoint_path),
                "sha256": self.checkpoint_sha256,
            },
            "action_contract": dict(self.action_contract.to_report()),
            "policy_contract": dict(self.policy_contract),
            "environment": {"name": self.env_name},
            "train_seed": self.train_seed,
        }
        if self.stream_contract is not None:
            identity["procedural_stream_contract"] = dict(self.stream_contract)
        return identity

    def lineage(self) -> dict[str, Any]:
        lineage: dict[str, Any] = {
            "report": {
                "path": str(self.report_path),
                "sha256": self.report_sha256,
            }
        }
        if self.lineage_report_path != self.report_path:
            lineage["trained_lineage_report"] = {
                "path": str(self.lineage_report_path),
                "sha256": self.lineage_report_sha256,
            }
        parent_path = self.raw_report.get("source_checkpoint")
        parent_sha = self.raw_report.get("source_checkpoint_sha256")
        if isinstance(parent_path, str) and isinstance(parent_sha, str):
            lineage["parent_checkpoint"] = {
                "path": str(Path(parent_path).resolve()),
                "sha256": parent_sha,
            }
        return lineage


def load_fixed_door_checkpoint_bundle(
    checkpoint: str | Path,
    report_path: str | Path,
) -> FixedDoorCheckpointBundle:
    return _load_fixed_door_checkpoint_bundle(
        checkpoint,
        report_path,
        active_reports=frozenset(),
    )


def _load_fixed_door_checkpoint_bundle(
    checkpoint: str | Path,
    report_path: str | Path,
    *,
    active_reports: frozenset[Path],
) -> FixedDoorCheckpointBundle:
    checkpoint_path = Path(checkpoint).resolve()
    evidence_path = Path(report_path).resolve()
    if evidence_path in active_reports:
        raise ValueError("fixed-door lineage contains a report cycle")
    report = _read_object(evidence_path)
    nested = report.get("trained_identity")
    if isinstance(nested, Mapping):
        _verify_evaluation_identity(report, evidence_path)
    identity = nested if isinstance(nested, Mapping) else report
    checkpoint_identity = identity.get("checkpoint")
    if isinstance(checkpoint_identity, Mapping):
        encoded_checkpoint_path = checkpoint_identity.get("path")
        encoded_checkpoint_sha = checkpoint_identity.get("sha256")
    else:
        encoded_checkpoint_path = identity.get("checkpoint")
        encoded_checkpoint_sha = identity.get("checkpoint_sha256")
    if not isinstance(encoded_checkpoint_path, str):
        raise ValueError("fixed-door report has no checkpoint path identity")
    if _identity_path(encoded_checkpoint_path) != checkpoint_path:
        raise ValueError("fixed-door checkpoint path does not match report")
    checkpoint_sha = _file_sha256(checkpoint_path)
    if encoded_checkpoint_sha != checkpoint_sha:
        raise ValueError("fixed-door checkpoint SHA-256 does not match report")

    encoded_action = identity.get("action_contract")
    if not isinstance(encoded_action, Mapping):
        raise ValueError("fixed-door report has no action contract")
    action_contract = approved_door_action_contract_from_report(encoded_action)
    encoded_policy = identity.get("policy_contract")
    if not isinstance(encoded_policy, Mapping):
        raise ValueError("fixed-door report has no policy contract")
    architecture = door_policy_architecture_from_report(encoded_policy)
    _verify_checkpoint_architecture(checkpoint_path, architecture)
    env_name = _environment_name(identity)
    train_seed = _train_seed(identity)
    stream = identity.get("procedural_stream_contract")
    if stream is not None:
        if not isinstance(stream, Mapping):
            raise ValueError("fixed-door procedural stream contract is invalid")
        verify_door_stream_contract(stream)

    lineage_path, lineage_sha = _lineage_identity(
        report,
        evidence_path,
    )
    bundle = FixedDoorCheckpointBundle(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        report_path=evidence_path,
        report_sha256=_file_sha256(evidence_path),
        action_contract=action_contract,
        policy_contract=dict(encoded_policy),
        architecture=architecture,
        env_name=env_name,
        train_seed=train_seed,
        stream_contract=None if stream is None else dict(stream),
        lineage_report_path=lineage_path,
        lineage_report_sha256=lineage_sha,
        raw_report=report,
    )
    if isinstance(nested, Mapping):
        lineage_bundle = _load_fixed_door_checkpoint_bundle(
            checkpoint_path,
            lineage_path,
            active_reports=active_reports | {evidence_path},
        )
        if bundle.trained_identity() != lineage_bundle.trained_identity():
            raise ValueError(
                "fixed-door trained identity does not match lineage report"
            )
    return bundle


def _environment_name(identity: Mapping[str, Any]) -> str:
    environment = identity.get("environment")
    config = identity.get("config")
    candidate = (
        environment.get("name")
        if isinstance(environment, Mapping)
        else config.get("env_name")
        if isinstance(config, Mapping)
        else FIXED_DOOR_ENV_NAME
    )
    if candidate != FIXED_DOOR_ENV_NAME:
        raise ValueError(f"unsupported fixed-door environment: {candidate!r}")
    return FIXED_DOOR_ENV_NAME


def _train_seed(identity: Mapping[str, Any]) -> int:
    config = identity.get("config")
    candidate = identity.get(
        "train_seed",
        config.get("seed", 11) if isinstance(config, Mapping) else 11,
    )
    try:
        seed = int(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid fixed-door training seed") from exc
    if seed < 0:
        raise ValueError("fixed-door training seed cannot be negative")
    return seed


def _lineage_identity(
    report: Mapping[str, Any],
    evidence_path: Path,
) -> tuple[Path, str]:
    lineage = report.get("lineage")
    encoded = lineage.get("report") if isinstance(lineage, Mapping) else None
    if not isinstance(encoded, Mapping):
        return evidence_path, _file_sha256(evidence_path)
    path = encoded.get("path")
    digest = encoded.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise ValueError("fixed-door lineage report identity is incomplete")
    resolved = _identity_path(path)
    if resolved == evidence_path:
        raise ValueError("fixed-door lineage report self-reference is invalid")
    if _file_sha256(resolved) != digest:
        raise ValueError("fixed-door lineage report SHA-256 does not match")
    return resolved, digest


def _identity_path(value: str) -> Path:
    return Path(value).resolve()


def _verify_evaluation_identity(
    report: Mapping[str, Any],
    evidence_path: Path,
) -> None:
    identity = report.get("evaluation_identity")
    encoded_report = (
        identity.get("report") if isinstance(identity, Mapping) else None
    )
    encoded_path = (
        encoded_report.get("path")
        if isinstance(encoded_report, Mapping)
        else None
    )
    if (
        not isinstance(identity, Mapping)
        or identity.get("kind") != "fixed_door_promotion"
        or identity.get("schema_version") != 1
        or not isinstance(encoded_path, str)
    ):
        raise ValueError("fixed-door evaluation identity is incomplete")
    if _identity_path(encoded_path) != evidence_path:
        raise ValueError("fixed-door evaluation report path does not match identity")


def _verify_checkpoint_architecture(
    checkpoint: Path,
    architecture: DoorPolicyArchitecture,
) -> None:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        fusion = state["encoder.fusion.0.weight"]
        recurrent = [
            state[f"network.layers.{index}.weight"]
            for index in range(architecture.num_layers)
        ]
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError("fixed-door checkpoint architecture is unreadable") from exc
    hidden = architecture.hidden_size
    if (
        int(fusion.shape[0]) != hidden
        or len(recurrent) != architecture.num_layers
        or any(tuple(weight.shape) != (3 * hidden, hidden) for weight in recurrent)
    ):
        raise ValueError(
            "fixed-door checkpoint architecture does not match policy contract"
        )
    unexpected_layer = f"network.layers.{architecture.num_layers}.weight"
    if unexpected_layer in state:
        raise ValueError(
            "fixed-door checkpoint architecture has unreported recurrent layers"
        )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("fixed-door report must contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
