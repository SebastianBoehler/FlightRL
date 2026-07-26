from __future__ import annotations

from pathlib import Path
import csv
import json
import subprocess
import sys

import numpy as np
import torch

from flightrl.sixdof.crash_replay import CrashReplayConfig, build_crash_replay_dataset, write_crash_replay_dataset
from flightrl.sixdof.crash_selection import crash_replay_selection_metrics, crash_replay_selection_score, load_replay_npz
from flightrl.sixdof.replay_loss import weighted_mse_loss


ROOT = Path(__file__).resolve().parents[1]


class FixedPolicy(torch.nn.Module):
    def __init__(self, actions: torch.Tensor) -> None:
        super().__init__()
        self.actions = actions

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actions[: observations.shape[0]]


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
        "range.front": 900.0,
        "range.back": 900.0,
        "range.left": 900.0,
        "range.right": 900.0,
        "range.up": 1500.0,
        "range.zrange": 500.0,
        "sys.canfly": 1.0,
        "sys.isTumbled": 0.0,
    }
    row.update(overrides)
    return row


def test_crash_replay_dataset_weights_precontact_rows() -> None:
    dataset = build_crash_replay_dataset(
        [
            telemetry_row(**{"stateEstimate.vx": 0.7}),
            telemetry_row(**{"range.front": 120.0}),
            telemetry_row(**{"sys.canfly": 0.0}),
        ],
        CrashReplayConfig(),
    )

    assert dataset["primary_groups"].tolist() == ["precontact_drift", "close_recovery", "unsafe_tail"]
    assert np.allclose(dataset["sample_weights"], [2.0, 1.0, 0.6])


def test_crash_replay_bounds_precontact_targets_independently() -> None:
    dataset = build_crash_replay_dataset(
        [telemetry_row(**{"stateEstimate.vx": 3.0})],
        CrashReplayConfig(precontact_target_clip_abs=0.25),
    )

    assert dataset["summary"]["group_counts"]["precontact_drift"] == 1
    assert float(np.max(np.abs(dataset["target_actions"]))) <= 0.25


def test_crash_replay_can_use_precontact_drift_brake_targets() -> None:
    row = telemetry_row(**{"stateEstimate.vx": 1.5})
    default = build_crash_replay_dataset([row], CrashReplayConfig())
    shaped = build_crash_replay_dataset(
        [row],
        CrashReplayConfig(target_shaping="precontact_drift_brake", target_shaping_strength=0.5),
    )

    assert shaped["config"]["target_shaping"] == "precontact_drift_brake"
    assert shaped["target_actions"][0, 2] < default["target_actions"][0, 2]


def test_weighted_crash_replay_npz_roundtrip(tmp_path: Path) -> None:
    output = tmp_path / "crash_replay.npz"
    dataset = build_crash_replay_dataset([telemetry_row(**{"stateEstimate.vx": 0.7})], CrashReplayConfig())

    write_crash_replay_dataset(output, dataset)
    replay = load_replay_npz(str(output))

    assert replay is not None
    assert torch.equal(replay["sample_weights"], torch.tensor([2.0]))
    assert replay["primary_groups"].tolist() == ["precontact_drift"]


def test_crash_replay_dataset_uses_logged_previous_action() -> None:
    rows = [
        telemetry_row(
            **{
                "stateEstimate.vx": 0.7,
                "action_thrust": 0.1,
                "action_roll_rate": 0.2,
                "action_pitch_rate": -0.3,
                "action_yaw_rate": 0.4,
            }
        ),
        telemetry_row(**{"stateEstimate.vx": 0.7}),
    ]

    dataset = build_crash_replay_dataset(rows, CrashReplayConfig())

    assert np.allclose(dataset["observations"][0, -4:], np.zeros(4))
    assert np.allclose(dataset["observations"][1, -4:], [0.1, 0.2, -0.3, 0.4])


def test_crash_selection_tracks_precontact_l2() -> None:
    replay = {
        "observations": torch.zeros(3, 28),
        "target_actions": torch.zeros(3, 4),
        "primary_groups": np.asarray(["precontact_drift", "close_recovery", "unsafe_tail"], dtype=object),
    }
    policy = FixedPolicy(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]]))

    metrics = crash_replay_selection_metrics(policy, replay, action_abs_limit=0.85)

    assert metrics["crash_replay_precontact_l2_p95"] > 0.9
    assert crash_replay_selection_score(metrics, action_abs_limit=0.85) < -0.4


def test_weighted_mse_loss_emphasizes_weighted_rows() -> None:
    prediction = torch.tensor([[1.0, 0.0], [0.1, 0.0]])
    target = torch.zeros(2, 2)

    unweighted = weighted_mse_loss(prediction, target, None)
    weighted = weighted_mse_loss(prediction, target, torch.tensor([2.0, 0.5]))

    assert weighted > unweighted


def test_crash_replay_builder_exposes_target_clip_knobs(tmp_path: Path) -> None:
    log = tmp_path / "crash.csv"
    output = tmp_path / "report.json"
    dataset = tmp_path / "dataset.npz"
    rows = [telemetry_row(**{"sys.canfly": 0.0, "range.front": 120.0})]
    with log.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_puffer_crash_replay_set.py"),
            "--log",
            f"crash:{log}",
            "--output",
            str(output),
            "--dataset-output",
            str(dataset),
            "--unsafe-target-clip-abs",
            "0.25",
            "--precontact-target-clip-abs",
            "0.35",
            "--precontact-weight",
            "3.0",
            "--target-shaping",
            "precontact_drift_brake",
            "--target-shaping-strength",
            "0.5",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text())
    saved = np.load(dataset, allow_pickle=True)
    assert report["config"]["unsafe_target_clip_abs"] == 0.25
    assert report["config"]["precontact_target_clip_abs"] == 0.35
    assert report["config"]["precontact_weight"] == 3.0
    assert report["config"]["target_shaping"] == "precontact_drift_brake"
    assert report["config"]["target_shaping_strength"] == 0.5
    assert float(np.max(np.abs(saved["target_actions"]))) <= 0.25
