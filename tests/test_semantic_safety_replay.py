from __future__ import annotations

import numpy as np
import torch

from flightrl.mujoco.semantic_safety_replay import (
    BalancedSafetyReplay,
    SafetyReplayConfig,
    action_corridor_clearance,
    safety_supervision_losses,
)


def test_action_corridor_clearance_matches_teacher_boundaries() -> None:
    ranges = np.array(
        [
            [0.60, 4.0, 4.0, 4.0],
            [4.00, 4.0, 0.45, 4.0],
            [4.00, 4.0, 1.00, 1.20],
        ],
        dtype=np.float32,
    )

    clearance = action_corridor_clearance(ranges)

    assert np.allclose(clearance, [0.60, 0.65, 1.20])


def test_replay_balances_danger_and_safe_sequences_with_burn_in() -> None:
    replay = BalancedSafetyReplay(
        vision_slice=slice(0, 4),
        vision_shape=(1, 2, 2),
        config=SafetyReplayConfig(
            capacity_per_class=2,
            samples_per_class=1,
            additions_per_class=2,
            burn_in_steps=1,
            replay_interval=1,
        ),
        seed=7,
    )
    observations = np.arange(4 * 3 * 6, dtype=np.float32).reshape(4, 3, 6)
    clearances = np.array(
        [
            [0.40, 0.50, 0.60],
            [0.50, 0.70, 0.80],
            [1.00, 1.10, 1.20],
            [0.70, 0.80, 0.85],
        ],
        dtype=np.float32,
    )
    resets = np.zeros((4, 3), dtype=np.float32)

    replay.add(observations, clearances, resets)
    batch = replay.sample(update=1)

    assert replay.counts == {"danger": 2, "safe": 1}
    assert batch is not None
    assert batch.vision.shape == (2, 3, 1, 2, 2)
    assert batch.vision.dtype == torch.float32
    assert batch.clearance_m.shape == (2, 3)
    assert batch.loss_mask.tolist() == [[0.0, 1.0, 1.0]] * 2
    assert sorted(batch.is_danger.tolist()) == [False, True]


def test_replay_safety_loss_keeps_gradient_for_bad_danger_prediction() -> None:
    predicted_clearance = torch.tensor(
        [[2.0, 2.0], [1.2, 1.2]],
        requires_grad=True,
    )
    target_clearance = torch.tensor([[0.4, 0.5], [1.5, 1.6]])
    mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    clearance_loss, risk_loss = safety_supervision_losses(
        predicted_clearance,
        target_clearance,
        mask,
    )
    (clearance_loss + risk_loss).backward()

    assert predicted_clearance.grad is not None
    assert float(predicted_clearance.grad[0, 1]) > 0.0
    assert torch.count_nonzero(predicted_clearance.grad[:, 0]) == 0


def test_clearance_loss_preserves_near_obstacle_emphasis() -> None:
    near_loss, _ = safety_supervision_losses(
        torch.tensor([0.8]),
        torch.tensor([0.4]),
    )
    safe_loss, _ = safety_supervision_losses(
        torch.tensor([1.8]),
        torch.tensor([1.4]),
    )

    assert torch.allclose(near_loss, 5.0 * safe_loss)
