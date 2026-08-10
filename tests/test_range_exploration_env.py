from __future__ import annotations

from math import pi

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from flightrl.exploration.range_batch import RangeExplorationBatch
from flightrl.exploration.range_env import RangeExplorationEnv
from flightrl.exploration.range_mapper import RangePose
from flightrl.exploration.range_safety import (
    RangeClearanceHold,
    shield_range_exploration_action,
)
from flightrl.exploration.range_world import RangeWorld


def test_generated_range_world_is_seeded_and_connected() -> None:
    first = RangeWorld.generate(301)
    repeated = RangeWorld.generate(301)
    different = RangeWorld.generate(302)

    assert np.array_equal(first.occupied, repeated.occupied)
    assert not np.array_equal(first.occupied, different.occupied)
    assert first.free_space_is_connected()
    assert different.free_space_is_connected()


def test_open_room_ray_range_matches_wall_geometry() -> None:
    world = RangeWorld.open_room()
    pose = RangePose(3.25, 3.25, 0.0)

    ranges, validity = world.horizontal_ranges(pose)

    assert ranges.tolist() == pytest.approx([3.05, 3.15, 3.05, 3.15], abs=0.03)
    assert validity.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_generated_start_pose_respects_initial_clearance_envelope() -> None:
    for seed in range(10):
        world = RangeWorld.generate(seed)
        pose = world.sample_pose(np.random.default_rng(seed))
        ranges, validity = world.horizontal_ranges(pose)
        finite = ranges[validity.astype(bool)]
        assert len(finite) == 0 or float(np.min(finite)) >= 0.35


def test_range_environment_passes_gymnasium_checker() -> None:
    env = RangeExplorationEnv(seed=401, maximum_episode_steps=40, stress=False)

    check_env(env, skip_render_check=True)


def test_safety_terminal_penalty_dominates_total_possible_coverage_reward() -> None:
    world = RangeWorld.open_room()
    env = RangeExplorationEnv(
        seed=402,
        maximum_episode_steps=40,
        stress=False,
        world=world,
        initial_pose=RangePose(0.25, 3.25, pi),
    )
    env.reset(seed=402)

    _obs, reward, terminated, truncated, info = env.step(
        np.asarray((1.0, 0.0), dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(-2.0)
    assert info["collision"] is False
    assert info["safety_terminal"] is True
    assert info["positive_reward_total"] <= 1.0


def test_front_clearance_veto_suppresses_forward_without_choosing_yaw() -> None:
    env = RangeExplorationEnv(
        seed=406,
        maximum_episode_steps=40,
        stress=False,
        world=RangeWorld.open_room(),
        initial_pose=RangePose(0.40, 3.25, pi),
    )
    env.reset(seed=406)
    before = env.truth_pose

    _obs, reward, terminated, truncated, info = env.step(
        np.asarray((1.0, 0.0), dtype=np.float32)
    )

    assert terminated is False
    assert truncated is False
    assert reward >= 0.0
    assert env.truth_pose.x_m == pytest.approx(before.x_m)
    assert env.truth_pose.y_m == pytest.approx(before.y_m)
    assert info["forward_clearance_override"] is True
    assert env.previous_action.tolist() == [0.0, 0.0]


def test_map_clearance_veto_catches_diagonal_obstacle_between_range_beams() -> None:
    map_crop = np.zeros((4, 32, 32), dtype=np.float32)
    map_crop[2, 14, 14] = 1.0

    shielded, emergency, reasons = shield_range_exploration_action(
        np.asarray((0.8, 0.7), dtype=np.float32),
        np.full(4, 1.5, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        map_crop,
    )

    assert shielded.tolist() == pytest.approx([0.0, 0.7])
    assert emergency is False
    assert reasons == ["estimated_map_clearance_override"]


def test_rear_mapped_obstacle_does_not_veto_forward_motion() -> None:
    map_crop = np.zeros((4, 32, 32), dtype=np.float32)
    map_crop[2, 18, 16] = 1.0

    shielded, emergency, reasons = shield_range_exploration_action(
        np.asarray((0.8, 0.0), dtype=np.float32),
        np.full(4, 1.5, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        map_crop,
    )

    assert shielded.tolist() == pytest.approx([0.8, 0.0])
    assert emergency is False
    assert reasons == []


def test_side_clearance_veto_suppresses_forward_during_policy_yaw() -> None:
    shielded, emergency, reasons = shield_range_exploration_action(
        np.asarray((0.8, -0.7), dtype=np.float32),
        np.asarray((1.5, 1.5, 0.30, 1.5), dtype=np.float32),
        np.ones(4, dtype=np.float32),
        np.zeros((4, 32, 32), dtype=np.float32),
    )

    assert shielded.tolist() == pytest.approx([0.0, -0.7])
    assert emergency is False
    assert reasons == ["horizontal_clearance_override"]


def test_clearance_hold_requires_ten_clear_steps_before_forward_resumes() -> None:
    hold = RangeClearanceHold()
    action = np.asarray((0.8, 0.6), dtype=np.float32)

    triggered, triggered_reasons = hold.apply(
        action,
        ["horizontal_clearance_override"],
    )
    held = [hold.apply(action, [])[0] for _ in range(10)]
    released, released_reasons = hold.apply(action, [])

    assert triggered.tolist() == pytest.approx([0.0, 0.6])
    assert triggered_reasons == ["horizontal_clearance_override"]
    assert all(value.tolist() == pytest.approx([0.0, 0.6]) for value in held)
    assert released.tolist() == pytest.approx([0.8, 0.6])
    assert released_reasons == []


def test_environment_applies_map_clearance_veto_without_overwriting_yaw() -> None:
    env = RangeExplorationEnv(seed=407, maximum_episode_steps=40, stress=False)
    env.reset(seed=407)
    env._last_observation[:4096].reshape(4, 32, 32)[2, 14, 14] = 1.0

    _obs, _reward, terminated, _truncated, info = env.step(
        np.asarray((0.8, 0.7), dtype=np.float32)
    )

    assert terminated is False
    assert info["forward_clearance_override"] is True
    assert env.previous_action.tolist() == pytest.approx([0.0, 0.7])


def test_positive_coverage_reward_cannot_exceed_one() -> None:
    env = RangeExplorationEnv(seed=403, maximum_episode_steps=200, stress=False)
    env.reset(seed=403)
    total_positive = 0.0

    for step in range(200):
        action = np.asarray((0.5, 0.8 if step % 20 < 10 else -0.8), dtype=np.float32)
        _obs, reward, terminated, truncated, _info = env.step(action)
        total_positive += max(0.0, reward)
        if terminated or truncated:
            break

    assert total_positive <= 1.0 + 1e-6


def test_batched_core_matches_single_environment_for_same_seed_and_actions() -> None:
    single = RangeExplorationEnv(seed=404, maximum_episode_steps=20, stress=False)
    batch = RangeExplorationBatch(
        num_envs=1,
        seed=404,
        maximum_episode_steps=20,
        stress=False,
    )
    single_obs, _ = single.reset(seed=404)
    batch.reset(seed=404)
    assert batch.observations[0] == pytest.approx(single_obs)

    for action in (
        np.asarray((0.4, 0.0), dtype=np.float32),
        np.asarray((0.2, 0.5), dtype=np.float32),
        np.asarray((0.0, -1.0), dtype=np.float32),
    ):
        expected = single.step(action)
        observed = batch.step(action[None, :])
        assert observed[0][0] == pytest.approx(expected[0])
        assert observed[1][0] == pytest.approx(expected[1])
        assert bool(observed[2][0]) is expected[2]
        assert bool(observed[3][0]) is expected[3]


def test_batched_reset_done_restarts_only_finished_environments() -> None:
    batch = RangeExplorationBatch(
        num_envs=2,
        seed=405,
        maximum_episode_steps=1,
        stress=False,
    )
    before = batch.observations.copy()
    terminal_obs, _reward, _terminated, truncated, _infos = batch.step(
        np.zeros((2, 2), dtype=np.float32)
    )
    assert truncated.tolist() == [True, True]

    reset = batch.reset_done(np.asarray((True, False)), seed=505)

    assert reset.shape == (2, 4106)
    assert not np.array_equal(reset[0], before[0])
    assert np.array_equal(reset[1], terminal_obs[1])
