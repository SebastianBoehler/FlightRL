from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from flightrl.puffer4_door_bundle import load_fixed_door_checkpoint_bundle
from flightrl.puffer4_door_contract import (
    CORRECTED_DOOR_ACTION_CONTRACT,
    LEGACY_V59_ACTION_CONTRACT,
)
from flightrl.puffer4_door_policy_contract import door_policy_contract_report
from flightrl.puffer4_door_runtime import DoorPufferRuntime


def _write_lineage(tmp_path: Path, *, hidden_size: int = 32) -> tuple[Path, Path]:
    checkpoint = tmp_path / "door.bin"
    torch.save(DoorPufferRuntime(hidden_size=hidden_size).state_dict(), checkpoint)
    report = tmp_path / "door.report.json"
    report.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "action_contract": CORRECTED_DOOR_ACTION_CONTRACT.to_report(),
                "policy_contract": door_policy_contract_report(
                    hidden_size=hidden_size,
                    num_layers=1,
                ),
                "config": {
                    "env_name": "flightrl_fixed_door_d1",
                    "seed": 23,
                },
            }
        )
    )
    return checkpoint, report


def test_bundle_loads_legacy_flat_lineage_and_decodes_architecture(
    tmp_path: Path,
) -> None:
    checkpoint, report = _write_lineage(tmp_path)

    bundle = load_fixed_door_checkpoint_bundle(checkpoint, report)

    assert bundle.checkpoint_path == checkpoint.resolve()
    assert bundle.report_path == report.resolve()
    assert bundle.architecture.hidden_size == 32
    assert bundle.architecture.num_layers == 1
    assert bundle.env_name == "flightrl_fixed_door_d1"
    assert bundle.train_seed == 23
    assert bundle.action_contract is CORRECTED_DOOR_ACTION_CONTRACT


def test_bundle_rejects_checkpoint_path_alias_even_with_matching_bytes(
    tmp_path: Path,
) -> None:
    checkpoint, report = _write_lineage(tmp_path)
    alias = tmp_path / "alias.bin"
    alias.write_bytes(checkpoint.read_bytes())

    with pytest.raises(ValueError, match="path"):
        load_fixed_door_checkpoint_bundle(alias, report)


def test_bundle_rejects_checkpoint_architecture_mismatch(tmp_path: Path) -> None:
    checkpoint, report = _write_lineage(tmp_path, hidden_size=32)
    payload = json.loads(report.read_text())
    payload["policy_contract"] = door_policy_contract_report(
        hidden_size=64,
        num_layers=1,
    )
    report.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="architecture"):
        load_fixed_door_checkpoint_bundle(checkpoint, report)


def test_bundle_loads_nested_promotion_identity(tmp_path: Path) -> None:
    checkpoint, lineage = _write_lineage(tmp_path)
    source = json.loads(lineage.read_text())
    evaluation = tmp_path / "door.promotion-evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "trained_identity": {
                    "checkpoint": {
                        "path": str(checkpoint.resolve()),
                        "sha256": source["checkpoint_sha256"],
                    },
                    "action_contract": source["action_contract"],
                    "policy_contract": source["policy_contract"],
                    "environment": {"name": "flightrl_fixed_door_d1"},
                    "train_seed": 23,
                },
                "evaluation_identity": {
                    "kind": "fixed_door_promotion",
                    "schema_version": 1,
                    "report": {"path": str(evaluation.resolve())},
                },
                "lineage": {
                    "report": {
                        "path": str(lineage.resolve()),
                        "sha256": hashlib.sha256(lineage.read_bytes()).hexdigest(),
                    }
                },
            }
        )
    )

    bundle = load_fixed_door_checkpoint_bundle(checkpoint, evaluation)

    assert bundle.architecture.hidden_size == 32
    assert bundle.lineage_report_path == lineage.resolve()
    assert bundle.lineage_report_sha256 == hashlib.sha256(
        lineage.read_bytes()
    ).hexdigest()

    copied = tmp_path / "copied-evaluation.json"
    copied.write_bytes(evaluation.read_bytes())
    with pytest.raises(ValueError, match="evaluation report path"):
        load_fixed_door_checkpoint_bundle(checkpoint, copied)

    recursive = json.loads(evaluation.read_text())
    recursive["lineage"]["report"] = {
        "path": str(evaluation.resolve()),
        "sha256": "self-reference-has-no-valid-digest",
    }
    evaluation.write_text(json.dumps(recursive))
    with pytest.raises(ValueError, match="self-reference"):
        load_fixed_door_checkpoint_bundle(checkpoint, evaluation)


def test_nested_bundle_rejects_trained_identity_not_in_lineage(
    tmp_path: Path,
) -> None:
    checkpoint, lineage = _write_lineage(tmp_path)
    source = json.loads(lineage.read_text())
    evaluation = tmp_path / "substituted.promotion-evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "trained_identity": {
                    "checkpoint": {
                        "path": str(checkpoint.resolve()),
                        "sha256": source["checkpoint_sha256"],
                    },
                    "action_contract": LEGACY_V59_ACTION_CONTRACT.to_report(),
                    "policy_contract": source["policy_contract"],
                    "environment": {"name": "flightrl_fixed_door_d1"},
                    "train_seed": 23,
                },
                "evaluation_identity": {
                    "kind": "fixed_door_promotion",
                    "schema_version": 1,
                    "report": {"path": str(evaluation.resolve())},
                },
                "lineage": {
                    "report": {
                        "path": str(lineage.resolve()),
                        "sha256": hashlib.sha256(lineage.read_bytes()).hexdigest(),
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="trained identity"):
        load_fixed_door_checkpoint_bundle(checkpoint, evaluation)


def test_checked_in_v59_annotated_reevaluation_is_bundle_compatible() -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoint = (
        root
        / "artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m"
        / "flightrl_fixed_door_d1_seed11_1048576.bin"
    )
    report = checkpoint.with_suffix(".reevaluation.json")

    bundle = load_fixed_door_checkpoint_bundle(checkpoint, report)

    assert bundle.checkpoint_sha256 == (
        "f676d12b9d37c27f4cc62f99beceec8f30e74c88be8564cb242c23755e202cce"
    )
    assert bundle.architecture.hidden_size == 96
