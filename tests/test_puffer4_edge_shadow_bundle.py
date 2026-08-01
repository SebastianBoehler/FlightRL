from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import pytest
import torch

from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_shadow_bundle import (
    EDGE_SHADOW_BUNDLE_SCHEMA,
    build_edge_shadow_bundle,
    load_edge_shadow_bundle,
    require_edge_shadow_bundle,
    write_edge_shadow_bundle,
)
from puffer4_edge_artifact_support import EdgeArtifacts
from puffer4_edge_shadow_support import shadow_artifacts


@pytest.fixture(autouse=True)
def _stable_puffer_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flightrl.puffer4_door_runner.require_clean_puffer_revision",
        lambda _root: {"git_commit": "a" * 40},
    )


def _bundle(tmp_path: Path) -> dict:
    artifacts = shadow_artifacts(tmp_path)
    return build_edge_shadow_bundle(
        checkpoint=artifacts.checkpoint,
        evaluation_report=_evaluation(artifacts),
        replay=_replay(artifacts),
    )


def test_offline_shadow_bundle_round_trip_is_typed_and_non_actuating(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    output = tmp_path / "shadow-bundle.json"

    write_edge_shadow_bundle(bundle, output)
    metadata = load_edge_shadow_bundle(output)

    assert bundle["schema"] == EDGE_SHADOW_BUNDLE_SCHEMA
    assert bundle["mode"] == "offline_passive_shadow"
    assert bundle["authority"] == "none"
    assert bundle["deployment_authority"] is False
    assert bundle["hardware_approved"] is False
    assert bundle["controls_drone"] is False
    assert metadata.hidden_size == 48
    assert metadata.trained_target_ids == (0,)
    assert metadata.checkpoint["sha256"]
    assert metadata.evaluation["sha256"]
    assert metadata.replay["sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authority", "shadow_control"),
        ("deployment_authority", True),
        ("deployment_authority", 0),
        ("hardware_approved", True),
        ("controls_drone", True),
    ],
)
def test_offline_shadow_bundle_cannot_be_forged_into_live_authority(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    bundle = _bundle(tmp_path)
    bundle[field] = value

    with pytest.raises(ValueError, match="never carry hardware authority"):
        require_edge_shadow_bundle(bundle)


@pytest.mark.parametrize(
    ("field", "filename"),
    [
        ("checkpoint_identity", "student.pt"),
        ("evaluation_identity", "held-out-evaluation.json"),
        ("replay_identity", "offline-replay.jsonl"),
    ],
)
def test_offline_shadow_bundle_rejects_changed_bound_files(
    tmp_path: Path,
    field: str,
    filename: str,
) -> None:
    bundle = _bundle(tmp_path)
    (tmp_path / filename).write_bytes(b"changed after bundling")

    with pytest.raises(ValueError, match="identity does not match"):
        require_edge_shadow_bundle(bundle)

    assert bundle[field]["sha256"]


def test_offline_shadow_bundle_rejects_checkpoint_metadata_forgery(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)

    wrong_hidden = deepcopy(bundle)
    wrong_hidden["hidden_size"] = 32
    with pytest.raises(ValueError, match="hidden size"):
        require_edge_shadow_bundle(wrong_hidden)

    wrong_targets = deepcopy(bundle)
    wrong_targets["trained_target_ids"] = [1]
    with pytest.raises(ValueError, match="target IDs"):
        require_edge_shadow_bundle(wrong_targets)

    wrong_contract = deepcopy(bundle)
    wrong_contract["policy_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="contract SHA-256"):
        require_edge_shadow_bundle(wrong_contract)


def test_offline_shadow_bundle_requires_passing_held_out_evaluation(
    tmp_path: Path,
) -> None:
    artifacts = shadow_artifacts(tmp_path)
    evaluation = _evaluation(artifacts)
    report = json.loads(evaluation.read_text())
    report["gate"] = {"passed": False, "failures": ["obstacle"]}
    evaluation.write_text(json.dumps(report) + "\n")

    with pytest.raises(ValueError, match="passing held-out"):
        build_edge_shadow_bundle(
            checkpoint=artifacts.checkpoint,
            evaluation_report=evaluation,
            replay=_replay(artifacts),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_profile", "profile"),
        ("agents", "agents"),
        ("steps", "steps"),
        ("seed", "seed"),
        ("profile", "profile"),
        ("extra_field", "fields"),
        ("source_identity", "source identity"),
        ("fingerprint", "native build"),
        ("metric", "metric fields"),
        ("scene_schema", "scene group schema"),
        ("legacy_schema", "scope"),
    ],
)
def test_offline_shadow_bundle_rejects_forged_evaluation_structure(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    artifacts = shadow_artifacts(tmp_path)
    evaluation = _evaluation(artifacts)
    report = json.loads(evaluation.read_text())
    if mutation == "missing_profile":
        report["profiles"].pop("obstacle")
    elif mutation == "legacy_schema":
        report["schema"] = "flightrl.edge_v3.closed_loop_evaluation.v1"
    elif mutation in {"agents", "steps", "seed"}:
        report["profiles"]["clean"][mutation] = 1
    elif mutation == "profile":
        report["profiles"]["mixed"]["profile"]["obstacle_probability"] = 0.0
    elif mutation == "extra_field":
        report["unverified"] = True
    elif mutation == "source_identity":
        report["source_identity"] = {}
    elif mutation == "metric":
        report["profiles"]["clean"]["metrics"].pop("low_light_episode_fraction")
    elif mutation == "scene_schema":
        report["profiles"]["clean"]["metrics"]["scene_group_schema_version"] = 2.0
    else:
        report["native_build_fingerprint"]["source_manifest_sha256"] = "f" * 64
    evaluation.write_text(json.dumps(report) + "\n")

    with pytest.raises(ValueError, match=match):
        build_edge_shadow_bundle(
            checkpoint=artifacts.checkpoint,
            evaluation_report=evaluation,
            replay=_replay(artifacts),
        )


@pytest.mark.parametrize("gate_mode", ("stale_pass", "forged_profile_pass"))
def test_offline_shadow_bundle_recomputes_profile_and_aggregate_gates(
    tmp_path: Path,
    gate_mode: str,
) -> None:
    artifacts = shadow_artifacts(tmp_path)
    evaluation = _evaluation(artifacts)
    report = json.loads(evaluation.read_text())
    clean = report["profiles"]["clean"]
    clean["metrics"]["action_rmse"] = 1.0
    if gate_mode == "forged_profile_pass":
        clean["gate"] = {"passed": True, "checks": {}, "failures": [], "thresholds": {}}
    evaluation.write_text(json.dumps(report) + "\n")

    with pytest.raises(ValueError, match="gate"):
        build_edge_shadow_bundle(
            checkpoint=artifacts.checkpoint,
            evaluation_report=evaluation,
            replay=_replay(artifacts),
        )


def test_offline_shadow_bundle_rejects_nonfinite_metrics_and_stale_native_sources(
    tmp_path: Path,
) -> None:
    artifacts = shadow_artifacts(tmp_path)
    evaluation = _evaluation(artifacts)
    report = json.loads(evaluation.read_text())
    report["profiles"]["clean"]["metrics"]["action_rmse"] = float("nan")
    evaluation.write_text(json.dumps(report) + "\n")
    with pytest.raises(ValueError, match="finite"):
        build_edge_shadow_bundle(
            checkpoint=artifacts.checkpoint,
            evaluation_report=evaluation,
            replay=_replay(artifacts),
        )

    report = json.loads(evaluation.read_text())
    report["profiles"]["clean"]["metrics"]["action_rmse"] = 0.05
    evaluation.write_text(json.dumps(report) + "\n")
    source = next(iter(report["native_build_fingerprint"]["source_files_sha256"]))
    Path(source).write_text("changed after evaluation\n")
    with pytest.raises(ValueError, match="source manifest"):
        build_edge_shadow_bundle(
            checkpoint=artifacts.checkpoint,
            evaluation_report=evaluation,
            replay=_replay(artifacts),
        )


def test_offline_shadow_bundle_rejects_replay_that_no_longer_reproduces(
    tmp_path: Path,
) -> None:
    artifacts = shadow_artifacts(tmp_path)
    replay = _replay(artifacts)
    records = replay.read_text().splitlines()
    step = json.loads(records[1])
    step["action_sha256"] = "0" * 64
    records[1] = json.dumps(step)
    replay.write_text("\n".join(records) + "\n")

    with pytest.raises(ValueError, match="does not reproduce"):
        build_edge_shadow_bundle(
            checkpoint=artifacts.checkpoint,
            evaluation_report=_evaluation(artifacts),
            replay=replay,
        )


def test_offline_shadow_bundle_rejects_raw_checkpoint_and_extra_fields(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.pt"
    torch.save(EdgeNavigationActor(hidden_size=48).state_dict(), raw)
    evaluation = tmp_path / "evaluation.json"
    replay = tmp_path / "replay.npz"
    evaluation.write_text("{}\n")
    replay.write_bytes(b"replay")

    with pytest.raises(ValueError, match="raw, partial, and legacy"):
        build_edge_shadow_bundle(
            checkpoint=raw,
            evaluation_report=evaluation,
            replay=replay,
        )

    bundle = _bundle(tmp_path)
    bundle["live_control"] = False
    with pytest.raises(ValueError, match="fields"):
        require_edge_shadow_bundle(bundle)


def _evaluation(artifacts: EdgeArtifacts) -> Path:
    assert artifacts.evaluation is not None
    return artifacts.evaluation


def _replay(artifacts: EdgeArtifacts) -> Path:
    assert artifacts.replay is not None
    return artifacts.replay
