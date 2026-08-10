from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from flightrl.exploration.range_checkpoint import (
    range_training_contract,
    save_range_checkpoint,
)
from flightrl.exploration.range_evaluation import evaluate_range_candidate
from flightrl.exploration.range_policy import RangeExplorationActorCritic
from flightrl.exploration.range_shadow import (
    range_replay_live_shadow_eligible,
    replay_range_shadow,
)


HEADER = (
    "host_time_s",
    "crazyflie_time_ms",
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.yaw",
    "pm.vbat",
    "pm.state",
    "stateEstimate.roll",
    "stateEstimate.pitch",
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "range.zrange",
    "motion.motion",
    "motion.squal",
)


def test_live_shadow_eligibility_requires_simulation_and_replay_passes() -> None:
    assert range_replay_live_shadow_eligible(
        simulation_gate_passed=True,
        replay_passed=True,
    ) is True
    assert range_replay_live_shadow_eligible(
        simulation_gate_passed=False,
        replay_passed=True,
    ) is False
    assert range_replay_live_shadow_eligible(
        simulation_gate_passed=True,
        replay_passed=False,
    ) is False


def _checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(901)
    model = RangeExplorationActorCritic(hidden_size=64)
    evaluation = evaluate_range_candidate(model, seeds=(901,), horizon=2)
    return save_range_checkpoint(
        tmp_path / "candidate.pt",
        model,
        evaluation,
        training=range_training_contract(
            seed=901,
            updates=0,
            num_envs=1,
            rollout_horizon=1,
            learning_rate=3e-4,
            action_std=0.25,
            frontier_aux_coef=0.0,
            shield_aux_coef=0.10,
            general_turn_commitment_coef=0.0,
            obstacle_turn_commitment_coef=0.10,
        ),
        source_revision="1ac9a0c1d63ab6e3781bf5cfd2c8873521d462fc",
    )


def _telemetry(path: Path, *, timestamps: tuple[int, ...], squal: int = 100) -> Path:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for index, timestamp in enumerate(timestamps):
            writer.writerow(
                {
                    "host_time_s": 1000.0 + index * 0.05,
                    "crazyflie_time_ms": timestamp,
                    "stateEstimate.x": index * 0.01,
                    "stateEstimate.y": 0.0,
                    "stateEstimate.z": 0.4,
                    "stateEstimate.yaw": 0.0,
                    "pm.vbat": 3.9,
                    "pm.state": 0,
                    "stateEstimate.roll": 0.0,
                    "stateEstimate.pitch": 0.0,
                    "stabilizer.roll": 0.0,
                    "stabilizer.pitch": 0.0,
                    "stabilizer.yaw": 0.0,
                    "range.front": 1500,
                    "range.back": 1500,
                    "range.left": 1000,
                    "range.right": 1000,
                    "range.up": 2000,
                    "range.zrange": 400,
                    "motion.motion": 176,
                    "motion.squal": squal,
                }
            )
    return path


def test_replay_shadow_logs_policy_and_shield_without_drone_control(tmp_path: Path) -> None:
    report = replay_range_shadow(
        _checkpoint(tmp_path),
        _telemetry(tmp_path / "telemetry.csv", timestamps=(100, 150, 200)),
        tmp_path / "shadow",
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "shadow" / "shadow_actions.jsonl").read_text().splitlines()
    ]
    assert report["controls_drone"] is False
    assert report["rows"] == 3
    assert report["active_rows"] == 3
    assert report["replay_passed"] is True
    assert len(records) == 3
    assert records[0]["crazyflie_time_ms"] == 100
    assert records[0]["raw_policy_action"] != records[0]["executed_action"]
    assert records[0]["executed_action"] == [0.0, 0.0]


def test_replay_shadow_rejects_out_of_order_device_time(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ordered"):
        replay_range_shadow(
            _checkpoint(tmp_path),
            _telemetry(tmp_path / "telemetry.csv", timestamps=(100, 90)),
            tmp_path / "shadow",
        )


def test_replay_shadow_fails_closed_on_low_flow_quality(tmp_path: Path) -> None:
    report = replay_range_shadow(
        _checkpoint(tmp_path),
        _telemetry(
            tmp_path / "telemetry.csv",
            timestamps=(100, 150),
            squal=20,
        ),
        tmp_path / "shadow",
    )

    assert report["replay_passed"] is False
    assert report["safety_reason_counts"]["low_flow_quality"] == 2
    assert report["authority"]["flight"] is False
