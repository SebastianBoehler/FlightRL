from __future__ import annotations

import torch

from flightrl.sixdof.puffer_policy import PufferPolicyMetadata
from flightrl.sixdof.transfer_selection import build_transfer_replay, transfer_shadow_selection_metrics, transfer_shadow_selection_score
from flightrl.sixdof.transfer_test import LiveLogCase, TransferTestConfig


class SaturatingPolicy(torch.nn.Module):
    metadata = PufferPolicyMetadata(observation_dim=28, hidden_size=16, action_dim=4, num_layers=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.ones((observations.shape[0], 4), dtype=torch.float32)


def row() -> dict[str, float]:
    return {
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


def test_transfer_selection_penalizes_raw_command_gate_failures() -> None:
    prepared = [(LiveLogCase("log", "unused.csv"), [row() for _ in range(120)])]

    metrics = transfer_shadow_selection_metrics(SaturatingPolicy(), prepared, TransferTestConfig(min_command_safe_rows=80))

    assert metrics["transfer_command_failure_count"] > 0.0
    assert metrics["transfer_command_saturation_excess"] > 0.0
    assert metrics["transfer_command_rate_excess"] > 0.0
    assert "safe_action_saturation" in metrics["transfer_shadow_labels"]["log"]["command_failures"]
    assert transfer_shadow_selection_score(metrics) < -10.0


def test_transfer_replay_uses_logged_previous_action_state() -> None:
    rows = [
        row()
        | {
            "action_thrust": 0.1,
            "action_roll_rate": -0.2,
            "action_pitch_rate": 0.3,
            "action_yaw_rate": -0.4,
        },
        row(),
    ]

    replay = build_transfer_replay([(LiveLogCase("log", "unused.csv"), rows)], TransferTestConfig(previous_action_observation_scale=1.0))

    assert replay is not None
    assert torch.allclose(replay["observations"][0, -4:], torch.zeros(4))
    assert torch.allclose(replay["observations"][1, -4:], torch.tensor([0.1, -0.2, 0.3, -0.4]))
    assert torch.equal(replay["sequence_start"], torch.tensor([True, False]))


def test_transfer_replay_scales_logged_previous_action_observation() -> None:
    rows = [
        row()
        | {
            "action_thrust": 0.1,
            "action_roll_rate": -0.2,
            "action_pitch_rate": 0.3,
            "action_yaw_rate": -0.4,
        },
        row(),
    ]

    replay = build_transfer_replay(
        [(LiveLogCase("log", "unused.csv"), rows)],
        TransferTestConfig(previous_action_observation_scale=0.25),
    )

    assert replay is not None
    assert torch.allclose(replay["observations"][1, -4:], torch.tensor([0.025, -0.05, 0.075, -0.1]))


def test_transfer_replay_marks_each_case_sequence_start() -> None:
    replay = build_transfer_replay(
        [
            (LiveLogCase("first", "unused.csv"), [row(), row()]),
            (LiveLogCase("second", "unused.csv"), [row()]),
        ],
        TransferTestConfig(),
    )

    assert replay is not None
    assert torch.equal(replay["sequence_start"], torch.tensor([True, False, True]))
