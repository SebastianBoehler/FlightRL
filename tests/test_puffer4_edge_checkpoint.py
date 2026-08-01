from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import pytest
import torch

from flightrl.evidence_scope import file_identity
from flightrl.puffer4_edge_collection_evidence import (
    canonical_edge_environment_config,
    edge_environment_config_sha256,
)
from flightrl.puffer4_edge_checkpoint import (
    EDGE_CHECKPOINT_FORMAT,
    EDGE_CHECKPOINT_SCHEMA,
    build_edge_checkpoint_payload,
    load_edge_checkpoint,
    require_edge_checkpoint,
    save_edge_checkpoint,
)
from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_sequence import (
    load_edge_sequence_dataset,
    write_edge_sequence_dataset,
)
from puffer4_edge_artifact_support import checkpoint_artifacts


def _payload(tmp_path: Path) -> dict:
    artifacts = checkpoint_artifacts(tmp_path)
    return torch.load(
        artifacts.checkpoint,
        map_location="cpu",
        weights_only=True,
    )


def test_edge_checkpoint_round_trip_is_exact_and_non_authoritative(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    output = tmp_path / "edge-v3.pt"

    save_edge_checkpoint(payload, output)
    actor, metadata = load_edge_checkpoint(output)

    assert payload["checkpoint_schema"] == EDGE_CHECKPOINT_SCHEMA
    assert payload["checkpoint_format"] == EDGE_CHECKPOINT_FORMAT
    assert payload["authority"] == "none"
    assert payload["deployment_authority"] is False
    assert payload["hardware_approved"] is False
    assert payload["controls_drone"] is False
    assert metadata.hidden_size == 48
    assert metadata.trained_target_ids == (0,)
    assert metadata.native_build_fingerprint == payload["native_build_fingerprint"]
    assert metadata.native_build_fingerprint["dependency_revision"] == {
        "git_commit": "a" * 40
    }
    assert metadata.dataset_identity["sha256"]
    assert metadata.training_identity["sha256"]
    for key, expected in payload["state_dict"].items():
        assert torch.equal(actor.state_dict()[key], expected)


def test_edge_loader_rejects_raw_partial_and_legacy_checkpoints(
    tmp_path: Path,
) -> None:
    actor = EdgeNavigationActor(hidden_size=48)
    raw = tmp_path / "raw.pt"
    torch.save(actor.state_dict(), raw)
    with pytest.raises(ValueError, match="raw, partial, and legacy"):
        load_edge_checkpoint(raw)

    payload = _payload(tmp_path)
    payload.pop("policy_contract_sha256")
    with pytest.raises(ValueError, match="raw, partial, and legacy"):
        require_edge_checkpoint(payload)

    payload = _payload(tmp_path)
    payload["checkpoint_schema"] = "flightrl.edge_v3.checkpoint.v2"
    with pytest.raises(ValueError, match="schema"):
        require_edge_checkpoint(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authority", "flight"),
        ("deployment_authority", True),
        ("deployment_authority", 0),
        ("hardware_approved", True),
        ("controls_drone", True),
    ],
)
def test_edge_checkpoint_cannot_claim_hardware_authority(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = _payload(tmp_path)
    payload[field] = value

    with pytest.raises(ValueError, match="non-authoritative"):
        require_edge_checkpoint(payload)


def test_edge_checkpoint_requires_exact_state_keys_shapes_dtypes_and_values(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    key = next(iter(payload["state_dict"]))

    missing = deepcopy(payload)
    missing["state_dict"].pop(key)
    with pytest.raises(ValueError, match="keys"):
        require_edge_checkpoint(missing)

    wrong_shape = deepcopy(payload)
    wrong_shape["state_dict"][key] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape or dtype"):
        require_edge_checkpoint(wrong_shape)

    wrong_dtype = deepcopy(payload)
    wrong_dtype["state_dict"][key] = wrong_dtype["state_dict"][key].double()
    with pytest.raises(ValueError, match="shape or dtype"):
        require_edge_checkpoint(wrong_dtype)

    nonfinite = deepcopy(payload)
    tensor = nonfinite["state_dict"][key]
    tensor.reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        require_edge_checkpoint(nonfinite)


@pytest.mark.parametrize("target_ids", [[], [True], [1, 1], [2, 0], [3]])
def test_edge_checkpoint_rejects_invalid_trained_target_ids(
    tmp_path: Path,
    target_ids: list[object],
) -> None:
    artifacts = checkpoint_artifacts(tmp_path)

    with pytest.raises((TypeError, ValueError)):
        build_edge_checkpoint_payload(
            EdgeNavigationActor(hidden_size=48),
            trained_target_ids=target_ids,
            dataset=artifacts.selection,
            training_report=artifacts.training,
        )


def test_edge_checkpoint_rejects_contract_and_source_identity_tampering(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    forged = deepcopy(payload)
    forged["policy_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="contract SHA-256"):
        require_edge_checkpoint(forged)

    output = tmp_path / "edge-v3.pt"
    save_edge_checkpoint(payload, output)
    Path(payload["dataset_identity"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity does not match"):
        load_edge_checkpoint(output)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("best_epoch", 1, "selected epoch"),
        ("best_selection_loss", 99.0, "selected epoch"),
        (
            "baseline_gate",
            {
                "passed": True,
                "checks": {"previous_action": False, "constant_grounding": True},
            },
            "baseline gate",
        ),
        ("baselines", {}, "baselines"),
        ("source_identity", {}, "source identity"),
    ],
)
def test_edge_checkpoint_rejects_forged_training_report_claims(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    payload = _payload(tmp_path)
    training = Path(payload["training_identity"]["path"])
    report = json.loads(training.read_text())
    report[field] = value

    _rebind_training(payload, training, report)

    with pytest.raises(ValueError, match=match):
        require_edge_checkpoint(payload)


def test_edge_checkpoint_rejects_nonfinite_or_inconsistent_epoch_metrics(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    training = Path(payload["training_identity"]["path"])
    report = json.loads(training.read_text())
    report["history"][1]["selection"]["decision_action_loss"] = float("nan")

    _rebind_training(payload, training, report)

    with pytest.raises(ValueError, match="finite training metrics"):
        require_edge_checkpoint(payload)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("selection_split", "canonical"),
        ("physical_seed", "disjoint"),
        ("appearance_seed", "disjoint"),
        ("native_fingerprint", "native build"),
    ],
)
def test_edge_checkpoint_rejects_mismatched_training_datasets(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    payload = _payload(tmp_path)
    training = Path(payload["training_identity"]["path"])
    report = json.loads(training.read_text())
    selection = load_edge_sequence_dataset(payload["dataset_identity"]["path"])
    train = load_edge_sequence_dataset(report["datasets"]["train"]["path"])
    if mutation == "selection_split":
        selection.metadata["split"] = "final"
    elif mutation == "physical_seed":
        selection.metadata["base_seed"] = train.metadata["base_seed"]
    elif mutation == "appearance_seed":
        selection.metadata["appearance_seed"] = train.metadata["appearance_seed"]
    else:
        selection.metadata["native_build_fingerprint"]["extension"]["sha256"] = (
            "d" * 64
        )
    if mutation in {"physical_seed", "appearance_seed"}:
        _rebind_environment_config(selection)
    selection_path = tmp_path / "mutated-selection.npz"
    write_edge_sequence_dataset(selection_path, selection)
    payload["dataset_identity"] = file_identity(selection_path)
    report["datasets"]["selection"] = file_identity(selection_path)
    _rebind_training(payload, training, report)

    with pytest.raises(ValueError, match=match):
        require_edge_checkpoint(payload)


@pytest.mark.parametrize(
    ("hidden_size", "target_ids", "match"),
    [
        (32, [0], "hidden size"),
        (48, [0, 2], "target"),
    ],
)
def test_edge_checkpoint_provenance_must_match_actor_contract_and_targets(
    tmp_path: Path,
    hidden_size: int,
    target_ids: list[int],
    match: str,
) -> None:
    artifacts = checkpoint_artifacts(tmp_path)

    with pytest.raises(ValueError, match=match):
        build_edge_checkpoint_payload(
            EdgeNavigationActor(hidden_size=hidden_size),
            trained_target_ids=target_ids,
            dataset=artifacts.selection,
            training_report=artifacts.training,
        )


def _rebind_training(payload: dict, path: Path, report: dict) -> None:
    path.write_text(json.dumps(report) + "\n")
    payload["training_identity"] = file_identity(path)


def _rebind_environment_config(dataset) -> None:
    metadata = dataset.metadata
    config = canonical_edge_environment_config(
        environment=metadata["environment"],
        agents=metadata["agents"],
        base_seed=metadata["base_seed"],
        appearance_seed=metadata["appearance_seed"],
        collection_profile=metadata["collection_profile"],
    )
    metadata["environment_config"] = config
    metadata["environment_config_sha256"] = edge_environment_config_sha256(config)
