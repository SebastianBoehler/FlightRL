from __future__ import annotations

import numpy as np
import torch

from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.env import euler_to_quat
from flightrl.sixdof.policies import teacher_actions
from flightrl.sixdof.rl import PpoConfig, SixDofActorCritic, collect_rollout, compute_advantages, position_error_for_task_indices, ppo_update, rollout_reward


def test_compute_advantages_shapes_match_rollout() -> None:
    rollout = {
        "rewards": np.ones((3, 2), dtype=np.float32),
        "dones": np.zeros((3, 2), dtype=np.float32),
        "values": np.zeros((3, 2), dtype=np.float32),
        "next_value": np.zeros(2, dtype=np.float32),
    }
    advantages, returns = compute_advantages(rollout, gamma=0.9, gae_lambda=0.95)
    assert advantages.shape == (6,)
    assert returns.shape == (6,)
    assert float(advantages.max()) > 1.0


def test_ppo_update_runs_on_short_rollout() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=3, reset_profile="position_yaw_easy")
    model = SixDofActorCritic(input_dim=28, hidden_size=16)
    rollout = collect_rollout(env, model, horizon=3, action_std=0.2)
    assert rollout["teacher_actions"].shape == rollout["actions"].shape
    assert rollout["pre_tanh_actions"].shape == rollout["actions"].shape
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    reference = SixDofActorCritic(input_dim=28, hidden_size=16).actor
    stats = ppo_update(model, optimizer, rollout, PpoConfig(hidden_size=16, minibatch_size=4, update_epochs=1, action_std=0.2, imitation_coef=0.1, reference_coef=0.2), reference)
    assert set(stats) == {"policy_loss", "value_loss", "entropy", "imitation_loss", "reference_loss"}


def test_collect_rollout_labels_teacher_on_recorded_state() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=5, reset_profile="position_yaw_easy")
    expected = teacher_actions(env, task=env.task)
    rollout = collect_rollout(env, SixDofActorCritic(input_dim=28, hidden_size=16), horizon=1, action_std=0.2)
    np.testing.assert_allclose(rollout["teacher_actions"][0], expected, rtol=1e-6, atol=1e-6)


def test_collect_rollout_teacher_residual_executes_teacher_and_trains_zero_residual() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=6, reset_profile="position_yaw_easy")
    expected = teacher_actions(env, task=env.task)
    rollout = collect_rollout(
        env,
        SixDofActorCritic(input_dim=28, hidden_size=16),
        horizon=1,
        action_std=0.2,
        controller="teacher_residual",
        residual_scale=0.0,
    )

    np.testing.assert_allclose(rollout["executed_actions"][0], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rollout["teacher_actions"][0], np.zeros_like(expected), rtol=1e-6, atol=1e-6)


def test_collect_rollout_supports_progress_reward() -> None:
    env_progress = SixDofCrazyflieEnv(num_envs=4, seed=7, reset_profile="position_yaw_easy")
    env_clearance = SixDofCrazyflieEnv(num_envs=4, seed=7, reset_profile="position_yaw_easy")
    env_live = SixDofCrazyflieEnv(num_envs=4, seed=7, reset_profile="position_yaw_easy")
    env_base = SixDofCrazyflieEnv(num_envs=4, seed=7, reset_profile="position_yaw_easy")
    model = SixDofActorCritic(input_dim=28, hidden_size=16)
    shaped = collect_rollout(env_progress, model, horizon=2, action_std=0.2, reward_mode="progress")
    clearance = collect_rollout(env_clearance, model, horizon=2, action_std=0.2, reward_mode="progress_clearance")
    live_clearance = collect_rollout(env_live, model, horizon=2, action_std=0.2, reward_mode="live_clearance")
    raw = collect_rollout(env_base, model, horizon=2, action_std=0.2, reward_mode="env")
    assert shaped["rewards"].shape == raw["rewards"].shape
    assert not np.allclose(shaped["rewards"], raw["rewards"])
    assert not np.allclose(clearance["rewards"], shaped["rewards"])
    assert not np.allclose(live_clearance["rewards"], clearance["rewards"])


def test_yaw_clearance_reward_penalizes_true_angle_error() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=9, reset_profile="position_yaw_easy")
    env.quaternion[:] = euler_to_quat(np.zeros(2), np.zeros(2), np.asarray([0.0, np.pi], dtype=np.float32))
    env.target_yaw[:] = 0.0
    env.velocity[:] = 0.0
    env._update_ranges()
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    actions = np.zeros((2, 4), dtype=np.float32)
    done = np.zeros(2, dtype=bool)

    clearance = rollout_reward(env, np.zeros(2, dtype=np.float32), done, previous_error, actions, "progress_clearance")
    yaw_clearance = rollout_reward(env, np.zeros(2, dtype=np.float32), done, previous_error, actions, "progress_yaw_clearance")

    assert yaw_clearance[0] == clearance[0]
    assert yaw_clearance[1] < clearance[1] - 1.0


def test_live_clearance_reward_penalizes_close_obstacles_more() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=9, reset_profile="position_yaw_easy")
    env.ranges_m[:, :4] = np.asarray([[0.2, 2.0, 2.0, 2.0], [0.8, 2.0, 2.0, 2.0]], dtype=np.float32)
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    actions = np.zeros((2, 4), dtype=np.float32)
    done = np.zeros(2, dtype=bool)

    clearance = rollout_reward(env, np.zeros(2, dtype=np.float32), done, previous_error, actions, "progress_clearance")
    live_clearance = rollout_reward(env, np.zeros(2, dtype=np.float32), done, previous_error, actions, "live_clearance")

    assert live_clearance[0] < clearance[0]
    assert live_clearance[1] > clearance[1]


def test_live_stable_clearance_penalizes_open_space_drift() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=9, reset_profile="position_yaw_easy")
    env.ranges_m[:, :4] = 1.2
    env.velocity[:] = np.asarray([[1.2, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    actions = np.zeros((2, 4), dtype=np.float32)
    done = np.zeros(2, dtype=bool)

    live_clearance = rollout_reward(env, np.zeros(2, dtype=np.float32), done, previous_error, actions, "live_clearance")
    stable = rollout_reward(env, np.zeros(2, dtype=np.float32), done, previous_error, actions, "live_stable_clearance")

    assert stable[0] < live_clearance[0] - 0.4
    assert stable[1] == live_clearance[1]


def test_circle_progress_reward_uses_orbit_error_not_center_error() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=10, task="circle", reset_profile="circle_recovery")
    env.position[:] = np.asarray([[0.75, 0.0, 0.65]], dtype=np.float32)
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.65]], dtype=np.float32)

    error = position_error_for_task_indices(env, ("circle",), np.zeros(1, dtype=np.int64))

    assert error[0] < 1e-5
    assert np.linalg.norm(env.target_position - env.position, axis=1)[0] > 0.7


def test_circle_reward_and_teacher_use_sampled_target_altitude() -> None:
    env = SixDofCrazyflieEnv(
        num_envs=1,
        seed=10,
        task="circle",
        reset_profile="circle_recovery",
    )
    env.position[:] = np.asarray([[0.75, 0.0, 0.65]], dtype=np.float32)
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.82]], dtype=np.float32)
    task_indices = np.zeros(1, dtype=np.int64)

    error_below_target = position_error_for_task_indices(
        env,
        ("circle",),
        task_indices,
    )
    teacher_below_target = teacher_actions(env, task="circle")
    env.position[:, 2] = env.target_position[:, 2]
    error_at_target = position_error_for_task_indices(
        env,
        ("circle",),
        task_indices,
    )

    assert np.allclose(error_below_target, [0.17])
    assert error_at_target[0] < 1e-6
    assert teacher_below_target[0, 0] > 0.02


def test_default_circle_reward_uses_orbit_error_and_tangent_yaw() -> None:
    env = SixDofCrazyflieEnv(
        num_envs=1,
        seed=10,
        task="circle",
        reset_profile="circle_recovery",
    )
    env.position[:] = np.asarray([[0.75, 0.0, 0.65]], dtype=np.float32)
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.65]], dtype=np.float32)
    env.quaternion[:] = euler_to_quat(
        np.zeros(1),
        np.zeros(1),
        np.asarray([np.pi / 2.0], dtype=np.float32),
    )
    env.target_yaw[:] = 0.0
    env.velocity[:] = 0.0
    env._update_ranges()

    reward = env._reward(np.zeros((1, 4), dtype=np.float32))

    assert reward[0] > 0.99


def test_collect_rollout_supports_history_observation_mode() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=11, reset_profile="position_yaw_easy")
    model = SixDofActorCritic(input_dim=60, hidden_size=16)
    rollout = collect_rollout(env, model, horizon=3, action_std=0.2, observation_mode="history1")

    assert rollout["observations"].shape == (3, 4, 60)
    assert rollout["teacher_actions"].shape == rollout["actions"].shape


def test_collect_rollout_supports_task_conditioned_observations() -> None:
    env = SixDofCrazyflieEnv(num_envs=6, seed=12, reset_profile="position_yaw_easy")
    model = SixDofActorCritic(input_dim=30, hidden_size=16)
    rollout = collect_rollout(env, model, horizon=3, action_std=0.2, tasks=("position_yaw", "obstacle_avoidance"), rng=np.random.default_rng(123))

    assert rollout["observations"].shape == (3, 6, 30)
    task_bits = rollout["observations"][:, :, -2:]
    np.testing.assert_allclose(np.sum(task_bits, axis=2), 1.0)
    assert np.any(task_bits[:, :, 0] == 1.0)
    assert np.any(task_bits[:, :, 1] == 1.0)


def test_collect_rollout_keeps_task_assignment_until_episode_reset() -> None:
    seed = 123
    tasks = ("position_yaw", "obstacle_avoidance")
    env = SixDofCrazyflieEnv(
        num_envs=6,
        seed=12,
        reset_profile="position_yaw_easy",
    )
    model = SixDofActorCritic(input_dim=30, hidden_size=16)

    rollout = collect_rollout(
        env,
        model,
        horizon=3,
        action_std=0.2,
        tasks=tasks,
        rng=np.random.default_rng(seed),
    )

    task_bits = rollout["observations"][:, :, -len(tasks) :]
    np.testing.assert_array_equal(task_bits[1], task_bits[0])
    np.testing.assert_array_equal(task_bits[2], task_bits[0])


def test_collect_rollout_applies_initial_task_yaw_context() -> None:
    tasks = ("position_yaw", "circle")
    env = SixDofCrazyflieEnv(num_envs=2, seed=12, task="position_yaw")
    env.position[:] = np.asarray(
        [[0.75, 0.0, 0.65], [0.75, 0.0, 0.65]],
        dtype=np.float32,
    )
    env.target_position[:] = np.asarray(
        [[0.0, 0.0, 0.65], [0.0, 0.0, 0.65]],
        dtype=np.float32,
    )
    env.target_yaw[:] = 0.0
    env.quaternion[:] = euler_to_quat(
        np.zeros(2),
        np.zeros(2),
        np.full(2, np.pi / 2.0, dtype=np.float32),
    )
    env.observations[:] = env.observation()

    rollout = collect_rollout(
        env,
        SixDofActorCritic(input_dim=30, hidden_size=16),
        horizon=1,
        action_std=0.2,
        tasks=tasks,
        rng=np.random.default_rng(1),
        task_probabilities=np.asarray([0.0, 1.0]),
    )

    assert np.all(np.abs(rollout["observations"][0, :, 16]) < 1e-5)
    assert np.all(np.abs(rollout["observations"][0, :, 17] - 1.0) < 1e-5)


def test_collect_rollout_resamples_only_when_episode_resets() -> None:
    seed = 321
    tasks = ("position_yaw", "obstacle_avoidance")
    env = SixDofCrazyflieEnv(
        num_envs=6,
        seed=13,
        reset_profile="position_yaw_easy",
    )
    env.step_count[:] = 799
    model = SixDofActorCritic(input_dim=30, hidden_size=16)

    rollout = collect_rollout(
        env,
        model,
        horizon=3,
        action_std=0.2,
        tasks=tasks,
        rng=np.random.default_rng(seed),
    )

    expected_rng = np.random.default_rng(seed)
    expected_initial = expected_rng.choice(len(tasks), size=6, p=(0.5, 0.5))
    expected_after_reset = expected_rng.choice(len(tasks), size=6, p=(0.5, 0.5))
    task_bits = rollout["observations"][:, :, -len(tasks) :]
    np.testing.assert_array_equal(np.argmax(task_bits[0], axis=1), expected_initial)
    np.testing.assert_array_equal(
        np.argmax(task_bits[1], axis=1),
        expected_after_reset,
    )
    np.testing.assert_array_equal(task_bits[2], task_bits[1])
