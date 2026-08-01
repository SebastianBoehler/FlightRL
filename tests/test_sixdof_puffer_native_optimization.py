from __future__ import annotations

import numpy as np

from flightrl.puffer4_sixdof_export import render_sixdof_puffer4_binding
from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.rl import SixDofActorCritic, collect_rollout, position_error_for_task_indices, rollout_reward


def test_native_reset_is_deterministic_and_within_room_bounds() -> None:
    left = SixDofCrazyflieEnv(num_envs=16, seed=1, use_native_step=True)
    right = SixDofCrazyflieEnv(num_envs=16, seed=2, use_native_step=True)

    left.native_reset(seed=123)
    right.native_reset(seed=123)

    np.testing.assert_allclose(left.position, right.position)
    np.testing.assert_allclose(left.target_position, right.target_position)
    assert np.all(left.position[:, 0] >= left.room.x_min)
    assert np.all(left.position[:, 0] <= left.room.x_max)
    assert np.all(left.position[:, 1] >= left.room.y_min)
    assert np.all(left.position[:, 1] <= left.room.y_max)
    assert np.all(left.position[:, 2] >= left.room.z_min)
    assert np.all(left.position[:, 2] <= left.room.z_max)
    assert np.all(left.ranges_m >= 0.0)


def test_native_circle_step_does_not_call_python_observation(monkeypatch) -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=7, task="circle", use_native_step=True)

    def fail_observation():
        raise AssertionError("native circle step must assemble observations in C")

    monkeypatch.setattr(env, "observation", fail_observation)
    actions = np.zeros((env.num_envs, 4), dtype=np.float32)

    obs, *_ = env.step(actions)

    assert obs.shape == (env.num_envs, 28)


def test_native_progress_reward_matches_python_reward() -> None:
    py_env = SixDofCrazyflieEnv(num_envs=8, seed=11, task="circle", reset_profile="circle_recovery", use_native_step=False)
    native_env = SixDofCrazyflieEnv(num_envs=8, seed=11, task="circle", reset_profile="circle_recovery", use_native_step=True)
    actions = np.zeros((8, 4), dtype=np.float32)
    task_indices = np.zeros(8, dtype=np.int32)
    previous_error = position_error_for_task_indices(py_env, ("circle",), task_indices)

    _, py_base, py_terminal, py_truncation, _ = py_env.step(actions)
    expected = rollout_reward(
        py_env,
        py_base,
        py_terminal | py_truncation,
        previous_error,
        actions,
        "progress_yaw_clearance",
        tasks=("circle",),
        task_indices=task_indices,
    )
    native_env.set_native_context(task_indices=task_indices, tasks=("circle",), reward_mode="progress_yaw_clearance", previous_error=previous_error)
    _, native_reward, *_ = native_env.step(actions)

    np.testing.assert_allclose(native_reward, expected, rtol=1e-5, atol=1e-5)


def test_native_default_circle_reward_matches_task_aware_python_reward() -> None:
    py_env = SixDofCrazyflieEnv(
        num_envs=8,
        seed=12,
        task="circle",
        reset_profile="circle_recovery",
        use_native_step=False,
    )
    native_env = SixDofCrazyflieEnv(
        num_envs=8,
        seed=12,
        task="circle",
        reset_profile="circle_recovery",
        use_native_step=True,
    )
    actions = np.zeros((8, 4), dtype=np.float32)

    _, py_reward, *_ = py_env.step(actions)
    _, native_reward, *_ = native_env.step(actions)

    np.testing.assert_allclose(native_reward, py_reward, rtol=1e-5, atol=1e-5)


def test_native_circle_reward_uses_sampled_target_altitude() -> None:
    env = SixDofCrazyflieEnv(
        num_envs=1,
        seed=13,
        task="circle",
        reset_profile="circle_recovery",
        use_native_step=True,
    )
    env.position[:] = np.asarray([[0.75, 0.0, 0.82]], dtype=np.float32)
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.82]], dtype=np.float32)
    env.velocity[:] = 0.0
    env.body_rates[:] = 0.0
    env.target_yaw[:] = np.pi / 2.0
    env.quaternion[:] = np.asarray([[0.70710677, 0.0, 0.0, 0.70710677]], dtype=np.float32)
    env._update_ranges()

    _, reward, *_ = env.step(np.zeros((1, 4), dtype=np.float32))

    assert reward[0] > 0.99


def test_native_live_clearance_reward_matches_python_reward() -> None:
    py_env = SixDofCrazyflieEnv(num_envs=8, seed=14, task="obstacle_avoidance", reset_profile="obstacle_close_live", use_native_step=False)
    native_env = SixDofCrazyflieEnv(num_envs=8, seed=14, task="obstacle_avoidance", reset_profile="obstacle_close_live", use_native_step=True)
    actions = np.zeros((8, 4), dtype=np.float32)
    task_indices = np.zeros(8, dtype=np.int32)
    previous_error = position_error_for_task_indices(py_env, ("obstacle_avoidance",), task_indices)

    _, py_base, py_terminal, py_truncation, _ = py_env.step(actions)
    expected = rollout_reward(
        py_env,
        py_base,
        py_terminal | py_truncation,
        previous_error,
        actions,
        "live_clearance",
        tasks=("obstacle_avoidance",),
        task_indices=task_indices,
    )
    native_env.set_native_context(task_indices=task_indices, tasks=("obstacle_avoidance",), reward_mode="live_clearance", previous_error=previous_error)
    _, native_reward, *_ = native_env.step(actions)

    np.testing.assert_allclose(native_reward, expected, rtol=1e-5, atol=1e-5)


def test_native_live_stable_clearance_reward_matches_python_reward() -> None:
    py_env = SixDofCrazyflieEnv(num_envs=8, seed=15, task="obstacle_avoidance", reset_profile="obstacle_close_live", use_native_step=False)
    native_env = SixDofCrazyflieEnv(num_envs=8, seed=15, task="obstacle_avoidance", reset_profile="obstacle_close_live", use_native_step=True)
    actions = np.zeros((8, 4), dtype=np.float32)
    task_indices = np.zeros(8, dtype=np.int32)
    previous_error = position_error_for_task_indices(py_env, ("obstacle_avoidance",), task_indices)

    _, py_base, py_terminal, py_truncation, _ = py_env.step(actions)
    expected = rollout_reward(
        py_env,
        py_base,
        py_terminal | py_truncation,
        previous_error,
        actions,
        "live_stable_clearance",
        tasks=("obstacle_avoidance",),
        task_indices=task_indices,
    )
    native_env.set_native_context(task_indices=task_indices, tasks=("obstacle_avoidance",), reward_mode="live_stable_clearance", previous_error=previous_error)
    _, native_reward, *_ = native_env.step(actions)

    np.testing.assert_allclose(native_reward, expected, rtol=1e-5, atol=1e-5)


def test_collect_rollout_uses_preallocated_buffers() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=13, reset_profile="position_yaw_easy", use_native_step=True)
    model = SixDofActorCritic(input_dim=28, hidden_size=16)

    rollout = collect_rollout(env, model, horizon=3, action_std=0.2)

    assert rollout["observations"].shape == (3, 4, 28)
    assert rollout["observations"].flags.c_contiguous
    assert rollout["actions"].flags.c_contiguous
    assert rollout["rewards"].flags.c_contiguous


def test_puffer_sixdof_binding_uses_shared_native_reset_core() -> None:
    binding = render_sixdof_puffer4_binding()

    assert "flightrl_sixdof_reset_one" in binding
    assert "env->position[0] = rnd(" not in binding
