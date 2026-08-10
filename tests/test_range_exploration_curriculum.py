from __future__ import annotations

import numpy as np
import torch

from flightrl.exploration.range_challenge_training import (
    RangeObstacleTrainingBatch,
)
from flightrl.exploration.range_challenge_evaluation import (
    range_obstacle_challenge_cases,
)
from flightrl.exploration.range_curriculum import (
    collect_range_natural_counterfactual_batch,
    sample_range_counterfactual_batch,
    train_range_counterfactual_curriculum,
    train_range_natural_curriculum,
)
from flightrl.exploration.range_evaluation import range_counterfactual_checks
from flightrl.exploration.range_policy import RangeExplorationActorCritic


def test_counterfactual_batch_pairs_mirrored_frontiers_and_blocked_ranges() -> None:
    observations, targets = sample_range_counterfactual_batch(seed=901, batch_size=8)

    assert observations.shape == (8, 4106)
    assert observations.dtype == np.float32
    assert targets.shape == (8, 2)
    maps = observations[:, :4096].reshape(-1, 4, 32, 32)
    assert np.array_equal(maps[0, :, :, ::-1], maps[1])
    assert targets[0, 1] == -targets[1, 1]
    assert targets[0, 0] == targets[1, 0]
    assert observations[2, 4096] < observations[0, 4096]
    assert targets[2, 0] == 0.0
    assert targets[0, 0] > 0.0


def test_counterfactual_curriculum_teaches_held_out_direction_and_stop() -> None:
    torch.manual_seed(902)
    model = RangeExplorationActorCritic(hidden_size=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    report = train_range_counterfactual_curriculum(
        model,
        optimizer,
        seed=902,
        steps=80,
        batch_size=64,
    )
    observations, _targets = sample_range_counterfactual_batch(seed=1902, batch_size=8)
    with torch.no_grad():
        actions, _value = model.forward_step(torch.from_numpy(observations))
    values = actions.numpy()

    assert report["final_loss"] < report["initial_loss"] * 0.35
    assert values[0, 1] > 0.05
    assert values[1, 1] < -0.05
    assert values[2, 0] + 0.05 < values[0, 0]
    assert values[3, 0] + 0.05 < values[1, 0]
    assert range_counterfactual_checks(model) == {
        "mirrored_frontier_direction": True,
        "front_obstacle_response": True,
    }


def test_natural_curriculum_uses_dense_mapper_states_and_exact_pairs() -> None:
    observations, targets = collect_range_natural_counterfactual_batch(
        seed=903,
        base_count=8,
    )

    assert observations.shape == (32, 4106)
    maps = observations[:, :4096].reshape(-1, 4, 32, 32)
    assert np.max(maps[::4, 1].sum(axis=(1, 2))) > 100
    assert np.max(maps[::4, 3].sum(axis=(1, 2))) > 8
    assert np.array_equal(maps[0, :, :, ::-1], maps[1])
    assert np.array_equal(maps[0], maps[2])
    assert targets[0, 1] == -targets[1, 1]
    assert targets[0, 0] == targets[1, 0] == 0.65
    assert targets[2, 0] == targets[3, 0] == 0.0
    assert observations[2, 4096] == 0.05


def test_natural_curriculum_generalizes_to_held_out_mapper_states() -> None:
    torch.manual_seed(904)
    model = RangeExplorationActorCritic(hidden_size=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    train_observations, train_targets = collect_range_natural_counterfactual_batch(
        seed=904,
        base_count=64,
    )
    heldout_observations, heldout_targets = collect_range_natural_counterfactual_batch(
        seed=1904,
        base_count=16,
    )

    report = train_range_natural_curriculum(
        model,
        optimizer,
        train_observations,
        train_targets,
        seed=904,
        steps=120,
        batch_size=64,
    )
    with torch.no_grad():
        actions, _value = model.forward_step(torch.from_numpy(heldout_observations))
    values = actions.numpy()
    clear = np.arange(len(values)) % 4 < 2
    blocked = ~clear

    assert report["source"] == "mapper_rollout"
    assert np.mean(np.sign(values[:, 1]) == np.sign(heldout_targets[:, 1])) > 0.75
    assert np.mean(np.abs(values[:, 1]) > 0.05) > 0.85
    assert values[clear, 0].mean() > values[blocked, 0].mean() + 0.40
    assert range_counterfactual_checks(model) == {
        "mirrored_frontier_direction": True,
        "front_obstacle_response": True,
    }


def test_obstacle_training_batch_varies_close_approaches_without_eval_geometry() -> None:
    batch = RangeObstacleTrainingBatch(num_envs=8, seed=905)
    evaluation_world = range_obstacle_challenge_cases()[0].world.occupied

    assert batch.observations.shape == (8, 4106)
    assert len(set(batch.approach_sides)) == 4
    for env, observation in zip(batch.envs, batch.observations, strict=True):
        assert 0.55 <= observation[4096] * 4.0 <= 0.75
        assert env.world.collides(env.truth_pose.x_m, env.truth_pose.y_m) is False
        assert np.array_equal(env.world.occupied, evaluation_world) is False
