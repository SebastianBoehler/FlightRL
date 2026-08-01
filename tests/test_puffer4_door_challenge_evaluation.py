from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from flightrl.puffer4_door_bundle import load_fixed_door_checkpoint_bundle
from flightrl.puffer4_door_challenge_evaluation import (
    load_matched_control_report,
)
from flightrl.puffer4_door_challenge_reporting import (
    build_door_challenge_report,
)
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)

def test_challenge_evaluation_module_exists() -> None:
    assert importlib.util.find_spec(
        "flightrl.puffer4_door_challenge_evaluation"
    ) is not None


def _trained_identity() -> dict:
    return {
        "checkpoint": {
            "path": "/tmp/door.bin",
            "sha256": "checkpoint-sha",
        },
        "action_contract": {"sha256": "action-sha"},
        "policy_contract": {"sha256": "policy-sha"},
        "environment": {"name": "flightrl_fixed_door_d1"},
        "train_seed": 11,
    }


def _metrics(success: float) -> dict:
    return {
        "status": "complete",
        "success_rate": success,
        "outside_fov_success_rate": success - 0.1,
        "collision_rate": 0.01,
        "finite_outputs": {"passed": True},
        "performance": {
            "closed_loop_agent_steps_per_second": 12_000.0,
            "closed_loop_batch_ms": {"p95": 4.0},
            "policy_forward_batch_ms": {"p95": 1.5},
        },
    }


def _write_control(
    path: Path,
    *,
    trained_identity: dict,
    native_fingerprint: dict,
    seed: int = 10_011,
    steps: int = 3_000,
    agents: int = 128,
) -> dict:
    report = {
        "evaluation_schema": "flightrl.fixed_door.promotion.v3",
        "trained_identity": json.loads(json.dumps(trained_identity)),
        "evaluation_identity": {
            "kind": "fixed_door_promotion",
            "schema_version": 1,
            "report": {"path": str(path.resolve())},
            "environment": {
                "name": "flightrl_fixed_door_d1",
                "seed": seed,
                "steps_per_condition": steps,
                "agents": agents,
            },
            "native_build_fingerprint": native_fingerprint,
            "action_contract_sha256": "action-sha",
            "policy_contract_sha256": "policy-sha",
            "procedural_stream_contract": {"contract_id": "stream"},
            "evidence_age_runtime_contract": (
                FIXED_DOOR_EVIDENCE_AGE_CONTRACT.to_report()
            ),
        },
        "full_camera": _metrics(0.8),
    }
    path.write_text(json.dumps(report))
    return report


def test_control_report_is_bound_by_path_sha_and_matched_identity(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "door.promotion-evaluation.json"
    trained = _trained_identity()
    native = {"extension_sha256": "native-sha"}
    _write_control(
        control_path,
        trained_identity=trained,
        native_fingerprint=native,
    )

    control = load_matched_control_report(
        control_path,
        trained_identity=trained,
        native_build_fingerprint=native,
        stream_contract={"contract_id": "stream"},
        seed=10_011,
        steps=3_000,
        agents=128,
    )

    assert control.path == control_path.resolve()
    assert control.sha256 == hashlib.sha256(
        control_path.read_bytes()
    ).hexdigest()
    assert control.metrics["success_rate"] == 0.8


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("trained_identity", "trained identity"),
        ("native_build_fingerprint", "native build fingerprint"),
        ("seed", "seed"),
        ("steps_per_condition", "steps"),
        ("agents", "agents"),
        ("report_path", "path"),
        ("stream_contract", "stream contract"),
        ("evidence_age_runtime_contract", "evidence-age contract"),
        ("missing_evidence_age_runtime_contract", "evidence-age contract"),
    ),
)
def test_control_report_rejects_unmatched_identity_or_budget(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    control_path = tmp_path / "door.promotion-evaluation.json"
    trained = _trained_identity()
    native = {"extension_sha256": "native-sha"}
    report = _write_control(
        control_path,
        trained_identity=trained,
        native_fingerprint=native,
    )
    if mutation == "trained_identity":
        report["trained_identity"]["checkpoint"]["sha256"] = "other"
    elif mutation == "native_build_fingerprint":
        report["evaluation_identity"]["native_build_fingerprint"] = {
            "extension_sha256": "other"
        }
    elif mutation == "report_path":
        report["evaluation_identity"]["report"]["path"] = str(
            tmp_path / "other.json"
        )
    elif mutation == "stream_contract":
        report["evaluation_identity"]["procedural_stream_contract"] = {
            "contract_id": "other"
        }
    elif mutation == "evidence_age_runtime_contract":
        report["evaluation_identity"]["evidence_age_runtime_contract"] = {
            "contract_id": "other"
        }
    elif mutation == "missing_evidence_age_runtime_contract":
        report["evaluation_identity"].pop("evidence_age_runtime_contract")
    else:
        report["evaluation_identity"]["environment"][mutation] += 1
    control_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match=message):
        load_matched_control_report(
            control_path,
            trained_identity=trained,
            native_build_fingerprint=native,
            stream_contract={"contract_id": "stream"},
            seed=10_011,
            steps=3_000,
            agents=128,
        )


def test_challenge_report_is_diagnostic_and_records_control_deltas(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.json"
    trained = _trained_identity()
    native = {"extension_sha256": "native-sha"}
    _write_control(
        control_path,
        trained_identity=trained,
        native_fingerprint=native,
    )
    control = load_matched_control_report(
        control_path,
        trained_identity=trained,
        native_build_fingerprint=native,
        stream_contract={"contract_id": "stream"},
        seed=10_011,
        steps=3_000,
        agents=128,
    )
    output = tmp_path / "challenge.json"

    report = build_door_challenge_report(
        output=output,
        trained_identity=trained,
        lineage={"report": {"path": "/tmp/train.json", "sha256": "train"}},
        native_build_fingerprint=native,
        stream_contract={"contract_id": "stream"},
        seed=10_011,
        steps=3_000,
        agents=128,
        challenge_spec={
            "name": "pixel-noise",
            "single_controlled_variable": "actor-input pixel noise",
            "combined_with_other_challenges": False,
            "limitation": "Detector evidence remains clean.",
        },
        metrics=_metrics(0.7),
        control=control,
    )

    assert report["trained_identity"] == trained
    assert report["evaluation_identity"]["report"]["path"] == str(
        output.resolve()
    )
    assert report["evaluation_identity"]["native_build_fingerprint"] == native
    assert report["evaluation_identity"]["evidence_age_runtime_contract"] == (
        FIXED_DOOR_EVIDENCE_AGE_CONTRACT.to_report()
    )
    assert report["matched_control"]["report"] == {
        "path": str(control_path.resolve()),
        "sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
    }
    assert report["comparison"]["delta_challenge_minus_control"][
        "success_rate"
    ] == pytest.approx(-0.1)
    assert report["challenge"]["resolved_single_variable"][
        "name"
    ] == "pixel-noise"
    assert report["simulation_gate"] is False
    assert report["live_eligible"] is False
    assert report["limitations"]


def test_challenge_report_cannot_be_loaded_as_promotion_lineage(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "challenge.json"
    report_path.write_text(
        json.dumps(
            {
                "trained_identity": {},
                "evaluation_identity": {
                    "kind": "fixed_door_challenge",
                    "schema_version": 1,
                    "report": {"path": str(report_path.resolve())},
                },
            }
        )
    )

    with pytest.raises(ValueError, match="evaluation identity"):
        load_fixed_door_checkpoint_bundle(tmp_path / "door.bin", report_path)
