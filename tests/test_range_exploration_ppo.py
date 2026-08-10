from __future__ import annotations

import numpy as np
import torch

from flightrl.exploration.range_batch import RangeExplorationBatch
from flightrl.exploration.range_policy import RangeExplorationActorCritic
from flightrl.exploration.range_ppo import (
    RangePpoConfig,
    collect_range_rollout,
    frontier_yaw_targets,
    range_ppo_update,
    turn_commitment_targets,
)


def test_map_memory_range_ppo_collects_real_rollout_and_updates_parameters() -> None:
    torch.manual_seed(503)
    env = RangeExplorationBatch(
        num_envs=2,
        seed=503,
        maximum_episode_steps=20,
        stress=False,
    )
    model = RangeExplorationActorCritic(hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    rollout = collect_range_rollout(
        env,
        model,
        horizon=4,
        action_std=0.2,
        reset_seed=1503,
    )
    metrics = range_ppo_update(
        model,
        optimizer,
        rollout,
        RangePpoConfig(update_epochs=2, action_std=0.2),
    )

    assert rollout["observations"].shape == (4, 2, 4106)
    assert rollout["actions"].shape == (4, 2, 2)
    assert np.all((0.0 <= rollout["actions"][..., 0]) & (rollout["actions"][..., 0] <= 1.0))
    assert np.all((-1.0 <= rollout["actions"][..., 1]) & (rollout["actions"][..., 1] <= 1.0))
    assert rollout["actor_observation_contains_truth"].item() == 0
    assert "initial_state" not in rollout
    assert "final_state" not in rollout
    assert rollout["executed_actions"].shape == (4, 2, 2)
    assert rollout["shield_overrides"].shape == (4, 2)
    assert all(np.isfinite(value) for value in metrics.values())
    assert {"frontier_aux_loss", "shield_aux_loss", "turn_commitment_loss"} <= set(metrics)
    assert any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, model.parameters(), strict=True)
    )


def test_frontier_centroid_target_respects_body_left_right_convention() -> None:
    observations = torch.zeros((3, 4106), dtype=torch.float32)
    maps = observations[:, :4096].reshape(3, 4, 32, 32)
    maps[0, 3, 12, 12] = 1.0
    maps[1, 3, 12, 20] = 1.0

    targets, active = frontier_yaw_targets(observations)

    assert targets[0] > 0.0
    assert targets[1] < 0.0
    assert active.tolist() == [True, True, False]


def test_turn_commitment_preserves_policy_chosen_yaw_only_while_blocked() -> None:
    observations = torch.zeros((4, 4106), dtype=torch.float32)
    observations[:, 4105] = torch.tensor((0.6, -0.7, 0.04, 0.8))
    shield_overrides = torch.tensor((True, True, True, False))

    targets, active = turn_commitment_targets(observations, shield_overrides)

    torch.testing.assert_close(targets, observations[:, 4105])
    assert active.tolist() == [True, True, False, False]
