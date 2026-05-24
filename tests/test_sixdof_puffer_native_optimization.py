from __future__ import annotations

import numpy as np

from flightrl.puffer4_sixdof_export import render_sixdof_puffer4_binding
from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.rl import PpoConfig, SixDofActorCritic, collect_rollout, position_error_for_task_indices, rollout_reward


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
