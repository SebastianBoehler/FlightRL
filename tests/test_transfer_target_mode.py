from __future__ import annotations

import numpy as np
import torch

from flightrl.sixdof.transfer_selection import build_transfer_replay
from flightrl.sixdof.transfer_test import LiveLogCase, TransferTestConfig, shadow_pairs


class CapturePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observation: torch.Tensor | None = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self.observation = observations.detach().clone()
        return torch.zeros((observations.shape[0], 4), dtype=torch.float32)


def row_at_offset() -> dict[str, float]:
    return {
        "stateEstimate.x": 2.0,
        "stateEstimate.y": -1.0,
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
    }


def test_shadow_pairs_current_pose_target_removes_origin_pull() -> None:
    policy = CapturePolicy()

    shadow_pairs(policy, [row_at_offset()], TransferTestConfig(target_mode="current_pose"))

    assert policy.observation is not None
    assert policy.observation[0, :3].numpy() == pytest_approx([0.0, 0.0, 0.0])


def test_shadow_pairs_fixed_origin_keeps_explicit_origin_pull() -> None:
    policy = CapturePolicy()

    shadow_pairs(policy, [row_at_offset()], TransferTestConfig(target_mode="fixed_origin"))

    assert policy.observation is not None
    assert policy.observation[0, :3].numpy() == pytest_approx([-1.0, 0.5, 0.0])


def test_transfer_replay_uses_latched_current_pose_target() -> None:
    replay = build_transfer_replay(
        [(LiveLogCase("offset", "unused.csv"), [row_at_offset()])],
        TransferTestConfig(target_mode="current_pose"),
    )

    assert replay is not None
    assert replay["observations"][0, :3].numpy() == pytest_approx([0.0, 0.0, 0.0])


def pytest_approx(values: list[float]):
    import pytest

    return pytest.approx(np.asarray(values, dtype=np.float32))
