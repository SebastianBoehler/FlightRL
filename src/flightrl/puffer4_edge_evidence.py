from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
from pathlib import Path

from flightrl.evidence_scope import file_identity, require_existing_file_identity
from flightrl.puffer4_edge_budget import edge_actor_budget
from flightrl.puffer4_edge_native_build import (
    require_matching_edge_native_build_fingerprints,
)
from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_sequence import (
    load_edge_sequence_dataset,
    require_disjoint_edge_dataset_structures,
)
from flightrl.puffer4_edge_training import (
    EDGE_LOSS_CONTRACT,
    EDGE_SELECTION_RULE,
    EDGE_TRAINING_REPORT_SCHEMA,
    EDGE_WEIGHTING_CONTRACT,
    EdgeTrainConfig,
    _evaluate_edge_sequence_loss,
    require_even_edge_tbptt_chunks,
)
from flightrl.puffer4_edge_training_data import edge_sequence_loss_weights
from flightrl.puffer4_edge_training_evidence import (
    require_training_selection_evidence,
)
from flightrl.puffer4_edge_training_state import edge_state_dict_sha256
from flightrl.puffer4_edge_training_sources import (
    EDGE_TRAINING_SOURCE_PATHS,
    ROOT as _TRAINING_ROOT,
)


ROOT = _TRAINING_ROOT


_AUTHORITY = {
    "authority": "none",
    "deployment_authority": False,
    "hardware_approved": False,
    "controls_drone": False,
}
_TRAINING_FIELDS = {
    "schema",
    "status",
    "selection_rule",
    "best_epoch",
    "best_selection_loss",
    "best_selection_metrics",
    "hidden_size",
    "trained_target_ids",
    "config",
    "loss_contract",
    "weighting_contract",
    "baselines",
    "baseline_gate",
    "policy_contract_sha256",
    "model_budget",
    "best_selection_visual_ablation_metrics",
    "selected_actor_state_sha256",
    "history",
    "datasets",
    "native_build_fingerprint",
    "source_identity",
    *_AUTHORITY,
}
def require_edge_training_evidence(
    path: str | Path,
    *,
    selection_identity: Mapping[str, str],
    hidden_size: int,
    trained_target_ids: Sequence[int],
    policy_contract_sha256: str,
    actor_state_dict: Mapping,
) -> dict:
    report = _read_mapping(path, "edge training report")
    if set(report) != _TRAINING_FIELDS:
        raise ValueError("edge training report fields are incompatible")
    if (
        report["schema"] != EDGE_TRAINING_REPORT_SCHEMA
        or report["status"] != "complete"
        or report["selection_rule"] != EDGE_SELECTION_RULE
    ):
        raise ValueError("edge training report schema or completion is incompatible")
    if report["hidden_size"] != hidden_size:
        raise ValueError("edge training hidden size does not match its checkpoint")
    if report["trained_target_ids"] != list(trained_target_ids):
        raise ValueError("edge training target IDs do not match its checkpoint")
    if report["policy_contract_sha256"] != policy_contract_sha256:
        raise ValueError("edge training contract does not match its checkpoint")
    _require_authority(report, "edge training report")
    require_source_identities(
        report["source_identity"], EDGE_TRAINING_SOURCE_PATHS, "edge training"
    )
    config = _require_config(report["config"])
    if report["loss_contract"] != EDGE_LOSS_CONTRACT:
        raise ValueError("edge training loss contract is incompatible")
    if report["weighting_contract"] != EDGE_WEIGHTING_CONTRACT:
        raise ValueError("edge training weighting contract is incompatible")
    actor = EdgeNavigationActor(hidden_size=hidden_size)
    expected_budget = edge_actor_budget(actor)
    if report["model_budget"] != expected_budget:
        raise ValueError("edge training model budget is incompatible")
    try:
        actor.load_state_dict(dict(actor_state_dict), strict=True)
        actor.eval()
        state_digest = edge_state_dict_sha256(actor.state_dict())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("edge training actor state is incompatible") from exc
    if report["selected_actor_state_sha256"] != state_digest:
        raise ValueError("edge training actor state digest does not match checkpoint")
    train, selection = _require_training_datasets(
        report["datasets"], selection_identity
    )
    require_even_edge_tbptt_chunks(train, config)
    expected_targets = list(trained_target_ids)
    for dataset in (train, selection):
        if (
            dataset.metadata["target_ids"] != expected_targets
            or dataset.metadata["policy_contract_sha256"] != policy_contract_sha256
        ):
            raise ValueError("edge training dataset target or contract is incompatible")
    if train.metadata["environment"] != selection.metadata["environment"]:
        raise ValueError("edge training datasets use different environments")
    fingerprint = require_matching_edge_native_build_fingerprints(
        report["native_build_fingerprint"],
        train.metadata["native_build_fingerprint"],
        selection.metadata["native_build_fingerprint"],
    )
    weights = edge_sequence_loss_weights(selection)
    require_training_selection_evidence(
        report,
        selection,
        config,
        actor,
        reproduce=lambda visual_ablation: _evaluate_edge_sequence_loss(
            actor,
            selection,
            config,
            weights,
            visual_ablation=visual_ablation,
        ),
    )
    return fingerprint


def require_source_identities(
    value: object,
    expected: Mapping[str, Path],
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{label} source identity fields are invalid")
    try:
        matches = all(
            value[name] == file_identity(path) for name, path in expected.items()
        )
    except OSError as exc:
        raise ValueError(f"{label} source identity is unavailable") from exc
    if not matches:
        raise ValueError(f"{label} source identity does not match current sources")


def _require_training_datasets(value: object, selection_identity):
    if not isinstance(value, Mapping) or set(value) != {"train", "selection"}:
        raise ValueError("edge training dataset identities are invalid")
    loaded = []
    for split in ("train", "selection"):
        identity = require_existing_file_identity(
            value[split], label=f"edge training {split} dataset"
        )
        loaded.append(load_edge_sequence_dataset(identity["path"]))
    if value["selection"] != selection_identity:
        raise ValueError("edge checkpoint selection dataset identity is inconsistent")
    require_disjoint_edge_dataset_structures(*loaded)
    return loaded


def _require_config(value: object) -> EdgeTrainConfig:
    if not isinstance(value, Mapping):
        raise ValueError("edge training config is invalid")
    try:
        config = EdgeTrainConfig(**dict(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("edge training config is invalid") from exc
    if dict(value) != asdict(config):
        raise ValueError("edge training config is not canonical")
    return config


def _read_mapping(path: str | Path, label: str) -> Mapping:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_authority(value: Mapping, label: str) -> None:
    if any(value.get(name) != expected for name, expected in _AUTHORITY.items()):
        raise ValueError(f"{label} must be explicitly non-authoritative")
