from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from flightrl.puffer4_door_contract import (
    CORRECTED_DOOR_ACTION_CONTRACT,
    LEGACY_V59_ACTION_CONTRACT,
)
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_door_policy_contract import door_policy_contract_report
from flightrl.puffer4_door_runtime import DoorPufferRuntime
from flightrl.puffer4_door_selection import (
    build_fixed_door_selection_report,
    write_exclusive_selection_report,
)
from flightrl.puffer4_door_stream_contract import door_stream_contract_report


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    success: float,
    *,
    outside: float,
    collision: float,
    worst: float,
    latency: float,
    throughput: float,
) -> dict:
    return {
        "status": "complete",
        "success_rate": success,
        "outside_fov_success_rate": outside,
        "collision_rate": collision,
        "requested_steps": 3_000,
        "completed_steps": 3_000,
        "finite_outputs": {"passed": True},
        "marginal_groups": {
            "status": "available",
            "worst_marginal_group": {
                "scope": "marginal_not_joint",
                "dimension": "layout_family",
                "category": 0,
                "support": 100,
                "successes": round(100 * worst),
                "conditional_success_rate": worst,
            },
        },
        "performance": {
            "batch_agents": 128,
            "policy_forward_batch_ms": {"p95": latency},
            "closed_loop_agent_steps_per_second": throughput,
        },
    }


def _screen_report(
    root: Path,
    *,
    seed: int,
    parent: Path,
    success: float,
) -> tuple[Path, Path]:
    directory = root / f"seed{seed}"
    directory.mkdir()
    checkpoint = directory / f"door_seed{seed}.bin"
    torch.save(DoorPufferRuntime(hidden_size=32).state_dict(), checkpoint)
    report = checkpoint.with_suffix(".report.json")
    report.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha(checkpoint),
                "source_checkpoint": str(parent.resolve()),
                "source_checkpoint_sha256": _sha(parent),
                "action_contract": CORRECTED_DOOR_ACTION_CONTRACT.to_report(),
                "policy_contract": door_policy_contract_report(
                    hidden_size=32,
                    num_layers=1,
                ),
                "procedural_stream_contract": door_stream_contract_report(),
                "selected_stage": "bootstrap",
                "config": {
                    "env_name": "flightrl_fixed_door_d1",
                    "agents": 128,
                    "horizon": 64,
                    "bootstrap_updates": 128,
                    "bootstrap_learning_rate": 0.001,
                    "bootstrap_max_policy_rollin": 0.0,
                    "fresh_control": True,
                    "rollouts": 0,
                    "screen_steps": 1_400,
                    "eval_steps": 11_000,
                    "seed": seed,
                    "evaluation_seed": 10_000 + seed,
                    "output_dir": str(directory),
                    "source_checkpoint": str(parent.resolve()),
                    "skip_build": True,
                },
                "evaluation": {
                    "evaluation_mode": "full_camera_and_masked_camera",
                    "full_camera": {
                        "success_rate": success,
                        "collision_rate": 0.01,
                    },
                    "masked_camera": {"success_rate": 0.01},
                },
            }
        )
    )
    return checkpoint, report


def _promotion(
    checkpoint: Path,
    lineage: Path,
    *,
    action_contract: dict,
    success: float,
    outside: float,
    collision: float,
    worst: float,
    latency: float,
    throughput: float,
    cap_success: float,
) -> Path:
    output = checkpoint.with_suffix(".promotion-evaluation.json")
    policy = door_policy_contract_report(hidden_size=32, num_layers=1)
    trained = {
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": _sha(checkpoint),
        },
        "action_contract": action_contract,
        "policy_contract": policy,
        "environment": {"name": "flightrl_fixed_door_d1"},
        "train_seed": 11,
    }
    source = json.loads(lineage.read_text())
    if "procedural_stream_contract" in source:
        trained["procedural_stream_contract"] = source[
            "procedural_stream_contract"
        ]
    full = _run(
        success,
        outside=outside,
        collision=collision,
        worst=worst,
        latency=latency,
        throughput=throughput,
    )
    reset = _run(
        success - 0.1,
        outside=outside - 0.1,
        collision=collision,
        worst=worst,
        latency=latency,
        throughput=throughput,
    )
    temporal = _run(
        success - 0.2,
        outside=outside - 0.2,
        collision=collision,
        worst=worst,
        latency=latency,
        throughput=throughput,
    )
    report = {
        "evaluation_schema": "flightrl.fixed_door.promotion.v3",
        "trained_identity": trained,
        "evaluation_identity": {
            "kind": "fixed_door_promotion",
            "schema_version": 1,
            "report": {"path": str(output.resolve())},
            "environment": {
                "name": "flightrl_fixed_door_d1",
                "seed": 20_011,
                "steps_per_condition": 3_000,
                "agents": 128,
            },
            "action_contract_sha256": action_contract["sha256"],
            "policy_contract_sha256": policy["sha256"],
            "procedural_stream_contract": door_stream_contract_report(),
            "evidence_age_runtime_contract": (
                FIXED_DOOR_EVIDENCE_AGE_CONTRACT.to_report()
            ),
            "native_build_fingerprint": {"manifest": "matched"},
        },
        "lineage": {
            "report": {"path": str(lineage.resolve()), "sha256": _sha(lineage)}
        },
        "full_camera": full,
        "masked_camera": _run(
            0.01,
            outside=0.0,
            collision=0.0,
            worst=0.0,
            latency=latency,
            throughput=throughput,
        ),
        "recurrence_reset_ablation": {
            "metrics": reset,
            "delta_vs_carried": {
                "success_rate": -0.1,
                "outside_fov_success_rate": -0.1,
                "collision_rate": 0.0,
            },
        },
        "temporal_order_ablation": {
            "metrics": temporal,
            "delta_vs_carried_ordered": {
                "success_rate": -0.2,
                "outside_fov_success_rate": -0.2,
                "collision_rate": 0.0,
            },
        },
        "live_yaw_cap_challenge": {
            "metrics": _run(
                cap_success,
                outside=0.66,
                collision=0.02,
                worst=worst,
                latency=latency,
                throughput=throughput,
            )
        },
    }
    output.write_text(json.dumps(report))
    return output


def _evidence(tmp_path: Path) -> dict:
    parent = tmp_path / "parent.bin"
    parent.write_bytes(b"parent")
    screens = {}
    for seed, success in ((11, 0.86), (23, 0.85), (47, 0.87)):
        screens[seed] = _screen_report(
            tmp_path,
            seed=seed,
            parent=parent,
            success=success,
        )
    candidate_checkpoint, candidate_lineage = screens[11]
    candidate_report = _promotion(
        candidate_checkpoint,
        candidate_lineage,
        action_contract=CORRECTED_DOOR_ACTION_CONTRACT.to_report(),
        success=0.86,
        outside=0.75,
        collision=0.01,
        worst=0.70,
        latency=1.1,
        throughput=900.0,
        cap_success=0.71,
    )
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline_checkpoint = baseline_dir / "v59.bin"
    torch.save(DoorPufferRuntime(hidden_size=32).state_dict(), baseline_checkpoint)
    baseline_lineage = baseline_checkpoint.with_suffix(".report.json")
    baseline_lineage.write_text(
        json.dumps(
            {
                "checkpoint": str(baseline_checkpoint.resolve()),
                "checkpoint_sha256": _sha(baseline_checkpoint),
                "source_checkpoint": str(parent.resolve()),
                "source_checkpoint_sha256": _sha(parent),
                "action_contract": LEGACY_V59_ACTION_CONTRACT.to_report(),
                "policy_contract": door_policy_contract_report(
                    hidden_size=32,
                    num_layers=1,
                ),
                "config": {
                    "env_name": "flightrl_fixed_door_d1",
                    "seed": 11,
                },
            }
        )
    )
    baseline_report = _promotion(
        baseline_checkpoint,
        baseline_lineage,
        action_contract=LEGACY_V59_ACTION_CONTRACT.to_report(),
        success=0.79,
        outside=0.74,
        collision=0.01,
        worst=0.69,
        latency=1.0,
        throughput=1_000.0,
        cap_success=0.60,
    )
    return {
        "candidate_checkpoint": candidate_checkpoint,
        "candidate_report": candidate_report,
        "baseline_checkpoint": baseline_checkpoint,
        "baseline_report": baseline_report,
        "screens": {seed: paths[1] for seed, paths in screens.items()},
    }


def test_selection_promotes_only_after_matched_complete_evidence(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)

    report = build_fixed_door_selection_report(**evidence)

    assert report["selection_passed"] is True
    assert report["recommended_checkpoint"]["sha256"] == _sha(
        evidence["candidate_checkpoint"]
    )
    assert report["next_gate"] == "shadow_only"
    assert report["live_cap_simulation_ready"] is True
    assert report["ablation_deltas"]["recurrence"]["success_rate"] == (
        pytest.approx(-0.1)
    )
    assert set(report["inputs"]["screens"]) == {"11", "23", "47"}


def test_selection_rejects_unmatched_promotion_runtime(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    candidate = json.loads(evidence["candidate_report"].read_text())
    candidate["evaluation_identity"]["environment"]["seed"] += 1
    evidence["candidate_report"].write_text(json.dumps(candidate))

    with pytest.raises(ValueError, match="evaluation environment"):
        build_fixed_door_selection_report(**evidence)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("native", "native build fingerprint"),
        ("masked_nonfinite", "masked evaluation"),
        ("missing_live_cap", "live-cap"),
    ),
)
def test_selection_requires_matched_complete_promotion_conditions(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    evidence = _evidence(tmp_path)
    candidate = json.loads(evidence["candidate_report"].read_text())
    if mutation == "native":
        candidate["evaluation_identity"]["native_build_fingerprint"] = {
            "manifest": "different"
        }
    elif mutation == "masked_nonfinite":
        candidate["masked_camera"]["finite_outputs"]["passed"] = False
    else:
        candidate.pop("live_yaw_cap_challenge")
    evidence["candidate_report"].write_text(json.dumps(candidate))

    with pytest.raises(ValueError, match=message):
        build_fixed_door_selection_report(**evidence)


def test_selection_rejects_unmatched_screen_budget(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    screen = evidence["screens"][47]
    payload = json.loads(screen.read_text())
    payload["config"]["horizon"] = 63
    screen.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="budget differs"):
        build_fixed_door_selection_report(**evidence)


def test_selection_failure_never_recommends_checkpoint(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    screen = evidence["screens"][23]
    payload = json.loads(screen.read_text())
    payload["evaluation"]["full_camera"]["success_rate"] = 0.83
    screen.write_text(json.dumps(payload))

    report = build_fixed_door_selection_report(**evidence)

    assert report["selection_passed"] is False
    assert "recommended_checkpoint" not in report
    assert report["next_gate"] == "shadow_only"


@pytest.mark.parametrize(
    ("metric", "value", "check_name"),
    (
        (
            "success_rate",
            0.04,
            "masked_camera_regression_at_most_0_02",
        ),
        (
            "collision_rate",
            0.025,
            "masked_collision_regression_at_most_0_02",
        ),
    ),
)
def test_masked_camera_regressions_are_lower_is_better(
    tmp_path: Path,
    metric: str,
    value: float,
    check_name: str,
) -> None:
    evidence = _evidence(tmp_path)
    candidate = json.loads(evidence["candidate_report"].read_text())
    candidate["masked_camera"][metric] = value
    evidence["candidate_report"].write_text(json.dumps(candidate))

    report = build_fixed_door_selection_report(**evidence)

    assert report["selection_checks"][check_name]["passed"] is False
    assert report["selection_passed"] is False
    assert "recommended_checkpoint" not in report


def test_selection_writer_is_exclusive_and_hash_bound(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    report = build_fixed_door_selection_report(**evidence)
    output = tmp_path / "selection.json"

    written = write_exclusive_selection_report(
        output,
        report,
        input_paths=(
            evidence["candidate_checkpoint"],
            evidence["candidate_report"],
            evidence["baseline_checkpoint"],
            evidence["baseline_report"],
            *evidence["screens"].values(),
        ),
    )

    assert json.loads(written.read_text()) == report
    with pytest.raises(FileExistsError):
        write_exclusive_selection_report(
            output,
            report,
            input_paths=(),
        )
    with pytest.raises(ValueError, match="alias"):
        write_exclusive_selection_report(
            evidence["candidate_report"],
            report,
            input_paths=(evidence["candidate_report"],),
        )
