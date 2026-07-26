from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy
from flightrl.hardware.sixdof_live_replay import live_env_from_telemetry, target_from_telemetry
from flightrl.sixdof import SixDofCrazyflieEnv, teacher_actions
from flightrl.sixdof.crash_replay import CrashReplayConfig, build_crash_replay_dataset, score_crash_replay_policy
from flightrl.sixdof.transfer_log_quality import SourceTeacherQualityConfig, score_source_teacher_quality
from flightrl.sixdof.transfer_test import (
    LiveLogCase,
    TransferTestConfig,
    raw_shadow_rows,
    shadow_gate,
    shadow_pairs,
    shadow_scored_pairs,
    sign_agreement,
)
from flightrl.sixdof.transfer_selection import build_transfer_replay


ROOT = Path(__file__).resolve().parents[1]

class ConstantPolicy(torch.nn.Module):
    metadata = PufferPolicyMetadata(observation_dim=28, hidden_size=16, action_dim=4, num_layers=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0.0, 0.1, -0.2, 0.0]], dtype=torch.float32).repeat(observations.shape[0], 1)


class PreviousFeedbackPolicy(torch.nn.Module):
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[:, -4:] + 0.25


class RangeSlotPolicy(torch.nn.Module):
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[:, 18:22]


def telemetry_row(**overrides: float) -> dict[str, float]:
    row = {
        "stateEstimate.x": 0.0,
        "stateEstimate.y": 0.0,
        "stateEstimate.z": 0.5,
        "stateEstimate.vx": 0.0,
        "stateEstimate.vy": 0.0,
        "stateEstimate.vz": 0.0,
        "stabilizer.roll": 0.0,
        "stabilizer.pitch": 0.0,
        "stabilizer.yaw": 0.0,
        "range.front": 800.0,
        "range.back": 900.0,
        "range.left": 700.0,
        "range.right": 600.0,
        "range.up": 1500.0,
        "range.zrange": 500.0,
        "sys.canfly": 1.0,
        "sys.isTumbled": 0.0,
    }
    row.update(overrides)
    return row


def test_raw_shadow_rows_emit_explicit_commander_pitch_sign() -> None:
    rows = raw_shadow_rows(ConstantPolicy(), [telemetry_row()], TransferTestConfig(min_command_safe_rows=1))

    assert rows[0]["pitch_rate_deg_s"] < 0.0
    assert rows[0]["commander_pitch_rate_deg_s"] == -rows[0]["pitch_rate_deg_s"]


def test_shadow_pairs_roll_forward_policy_previous_action() -> None:
    pairs = shadow_pairs(PreviousFeedbackPolicy(), [telemetry_row(), telemetry_row()], TransferTestConfig(previous_action_observation_scale=1.0))

    assert np.allclose(pairs[0][0], np.full(4, 0.25, dtype=np.float32)) and np.allclose(pairs[1][0], np.full(4, 0.50, dtype=np.float32))


def test_shadow_pairs_can_mask_previous_action_observation() -> None:
    config = TransferTestConfig(previous_action_observation_scale=0.0)

    pairs = shadow_pairs(PreviousFeedbackPolicy(), [telemetry_row(), telemetry_row()], config)

    assert np.allclose(pairs[0][0], np.full(4, 0.25, dtype=np.float32)) and np.allclose(pairs[1][0], np.full(4, 0.25, dtype=np.float32))


def test_shadow_pairs_honor_deckless_sensor_profile() -> None:
    pairs = shadow_pairs(RangeSlotPolicy(), [telemetry_row(**{"range.front": 500.0})], TransferTestConfig(sensor_profile="deckless"))

    assert np.allclose(pairs[0][0], np.ones(4, dtype=np.float32))


def test_transfer_replay_honors_target_yaw_config() -> None:
    prepared = [(LiveLogCase("yaw", "unused.csv"), [telemetry_row()])]

    zero_yaw = build_transfer_replay(prepared, TransferTestConfig(target_yaw_deg=0.0))
    quarter_turn = build_transfer_replay(prepared, TransferTestConfig(target_yaw_deg=90.0))

    assert zero_yaw is not None
    assert quarter_turn is not None
    assert not torch.equal(zero_yaw["observations"], quarter_turn["observations"])
    assert not torch.equal(zero_yaw["target_actions"], quarter_turn["target_actions"])


def test_transfer_replay_honors_deckless_sensor_profile() -> None:
    prepared = [(LiveLogCase("deckless", "unused.csv"), [telemetry_row(**{"range.front": 500.0})])]

    replay = build_transfer_replay(prepared, TransferTestConfig(sensor_profile="deckless"))

    assert replay is not None
    assert torch.allclose(replay["observations"][0, 18:24], torch.ones(6))


def test_transfer_replay_marks_vertical_rows() -> None:
    prepared = [(LiveLogCase("vertical_case", "unused.csv"), [telemetry_row()])]

    replay = build_transfer_replay(prepared, TransferTestConfig())

    assert replay is not None
    assert bool(replay["vertical_mask"][0])


def test_shadow_gate_flags_poor_live_teacher_alignment() -> None:
    groups = {
        "all": {
            "samples": 120,
            "l2_p95": 0.7,
            "action_abs_max": 0.4,
            "sign_agreement": {"thrust": 1.0, "roll_rate": 0.2, "pitch_rate": 0.8},
        },
        "close_lt_32cm": {"samples": 0},
    }

    gate = shadow_gate(groups, TransferTestConfig())

    assert gate["passed"] is False
    assert "shadow_l2_p95" in gate["failures"]
    assert "shadow_roll_rate_sign" in gate["failures"]


def test_failed_source_shadow_excludes_crash_replay_rows() -> None:
    pairs = [
        (np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32), telemetry_row()),
        (
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            telemetry_row(**{"stateEstimate.vx": 0.7}),
        ),
    ]

    scored, excluded = shadow_scored_pairs(
        pairs,
        LiveLogCase("failed", "unused.csv", failed_source=True),
        TransferTestConfig(),
    )
    clean_scored, clean_excluded = shadow_scored_pairs(
        pairs,
        LiveLogCase("clean", "unused.csv", failed_source=False),
        TransferTestConfig(),
    )

    assert len(scored) == 1
    assert len(excluded) == 1
    assert len(clean_scored) == 2
    assert clean_excluded == []


def test_transfer_replay_filters_failed_source_rows() -> None:
    prepared = [
        (
            LiveLogCase("failed", "unused.csv", failed_source=True),
            [telemetry_row(), telemetry_row(**{"stateEstimate.vx": 0.7})],
        )
    ]

    replay = build_transfer_replay(prepared, TransferTestConfig())

    assert replay is not None
    assert replay["observations"].shape[0] == 1
    assert replay["source_rows"] == 2
    assert replay["excluded_source_rows"] == 1
    assert torch.equal(replay["sample_weights"], torch.ones(1))


def test_transfer_replay_balances_case_weights() -> None:
    prepared = [
        (LiveLogCase("long", "unused.csv"), [telemetry_row(), telemetry_row(), telemetry_row()]),
        (LiveLogCase("short", "unused.csv"), [telemetry_row()]),
    ]

    replay = build_transfer_replay(prepared, TransferTestConfig())

    assert replay is not None
    weights = replay["sample_weights"].numpy()
    assert np.isclose(weights[:3].sum(), weights[3:].sum())
    assert weights[-1] > weights[0]


def test_sign_agreement_ignores_near_zero_targets() -> None:
    actual = np.asarray([-1.0, 1.0, 1.0], dtype=np.float32)
    expected = np.asarray([0.001, 0.03, -0.04], dtype=np.float32)

    assert sign_agreement(actual, expected, min_abs=0.02) == 0.5
    assert sign_agreement(actual, expected, min_abs=1e-4) == 1.0 / 3.0


def test_source_teacher_quality_detects_task_mismatched_logged_actions() -> None:
    row = telemetry_row(**{"range.front": 200.0})
    env = SixDofCrazyflieEnv(num_envs=1, seed=0, task="obstacle_avoidance")
    target = np.asarray([0.0, 0.0, 0.5], dtype=np.float32)
    live_env_from_telemetry(env, row, target=target_from_telemetry(row, target), target_yaw=0.0)
    teacher = teacher_actions(env, task="obstacle_avoidance")[0]
    aligned = {**row, **dict(zip(("action_thrust", "action_roll_rate", "action_pitch_rate", "action_yaw_rate"), teacher))}
    inverted = {**row, **dict(zip(("action_thrust", "action_roll_rate", "action_pitch_rate", "action_yaw_rate"), -teacher))}

    good = score_source_teacher_quality([aligned] * 25, SourceTeacherQualityConfig())
    bad = score_source_teacher_quality([inverted] * 25, SourceTeacherQualityConfig())

    assert good["gate"]["passed"] is True
    assert bad["gate"]["passed"] is False
    assert "source_teacher_pitch_rate_sign" in bad["gate"]["failures"]


def test_crash_replay_dataset_records_precontact_drift_and_commander_sign() -> None:
    dataset = build_crash_replay_dataset(
        [telemetry_row(**{"stateEstimate.vx": 0.7, "range.front": 900.0})],
        CrashReplayConfig(min_samples=1),
    )

    assert dataset["summary"]["group_counts"]["precontact_drift"] == 1
    assert dataset["observations"].shape == (1, 28)
    assert dataset["target_actions"].shape == (1, 4)
    assert dataset["teacher_setpoints"].shape == (1, 5)
    assert dataset["teacher_setpoints"][0, 3] == -dataset["teacher_setpoints"][0, 2]


def test_crash_replay_bounds_unsafe_tail_targets() -> None:
    dataset = build_crash_replay_dataset(
        [telemetry_row(**{"stateEstimate.vx": 0.7, "sys.canfly": 0.0})],
        CrashReplayConfig(min_samples=1, unsafe_target_clip_abs=0.25),
    )

    assert dataset["summary"]["group_counts"]["unsafe_tail"] == 1
    assert float(dataset["target_actions"].max()) <= 0.25
    assert float(dataset["target_actions"].min()) >= -0.25


def test_crash_replay_gate_flags_candidate_gap() -> None:
    report = score_crash_replay_policy(
        ConstantPolicy(),
        [telemetry_row(**{"stateEstimate.vx": 0.7}) for _ in range(4)],
        CrashReplayConfig(min_samples=1, min_group_samples=1, max_l2_p95=0.0, max_action_abs=0.05),
    )

    assert report["gate"]["passed"] is False
    assert "crash_l2_p95" in report["gate"]["failures"]
    assert "crash_action_abs" in report["gate"]["failures"]


def test_transfer_test_cli_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.bin"
    log = tmp_path / "log.csv"
    physics = tmp_path / "physics.json"
    sensor = tmp_path / "sensor.json"
    torch.save(PufferSixDofPolicy(PufferPolicyMetadata(28, 16, 4, 1)).state_dict(), checkpoint)
    physics.write_text(json.dumps({"physics_profile": {"mass_kg": 0.036, "linear_drag": 0.04}}))
    sensor.write_text(json.dumps({"sensor_profile": {"range_noise_std_m": 0.001, "action_lag_s": 0.01}}))
    with log.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(telemetry_row().keys()))
        writer.writeheader()
        for _ in range(4):
            writer.writerow(telemetry_row())

    output = tmp_path / "transfer.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_puffer_transfer_test_set.py"),
            "--candidate",
            f"tiny:{checkpoint}",
            "--live-log",
            f"smoke:{log}",
            "--output",
            str(output),
            "--steps",
            "2",
            "--num-envs",
            "4",
            "--physics-profile",
            str(physics),
            "--sensor-profile",
            str(sensor),
            "--domain-randomization",
            "crazyflie_training",
            "--crash-target-shaping", "precontact_drift_brake",
            "--crash-target-shaping-strength", "0.25",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "puffer_transfer_test=" in result.stdout
    assert output.exists()
    assert output.with_suffix(".md").exists()
    report = json.loads(output.read_text())
    assert (report["config"]["physics_profile"], report["config"]["sensor_profile"], report["config"]["domain_randomization"]) == (str(physics), str(sensor), "crazyflie_training")
    assert (report["config"]["crash_target_shaping"], report["config"]["crash_target_shaping_strength"]) == ("precontact_drift_brake", 0.25)
    assert report["source_quality_passed"] is True
    assert report["source_teacher_quality"]["smoke"]["samples"] == 0
    tiny_config = report["candidates"]["tiny"]["config"]
    assert (tiny_config["physics_profile"], tiny_config["sensor_profile"], tiny_config["domain_randomization"]) == (str(physics), str(sensor), "crazyflie_training")
