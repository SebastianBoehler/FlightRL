from __future__ import annotations

import numpy as np

from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.env import euler_to_quat
from flightrl.sixdof.rl import rollout_reward
from flightrl.sixdof.yaw import circle_tangent_yaw, yaw_error_for_task


def test_circle_yaw_reference_uses_tangent_not_reset_target_yaw() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=23)
    env.position[:] = np.asarray([[0.75, 0.0, 0.65]], dtype=np.float32)
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.65]], dtype=np.float32)
    env.target_yaw[:] = 0.0
    env.quaternion[:] = euler_to_quat(np.zeros(1), np.zeros(1), np.asarray([np.pi / 2], dtype=np.float32))

    assert np.allclose(circle_tangent_yaw(env), [np.pi / 2])
    assert yaw_error_for_task(env, "circle")[0] < 1e-5
    assert yaw_error_for_task(env, "position_yaw")[0] > 1.5


def test_circle_observation_encodes_tangent_yaw_error() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=31, task="circle")
    env.position[:] = np.asarray([[0.75, 0.0, 0.65]], dtype=np.float32)
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.65]], dtype=np.float32)
    env.target_yaw[:] = 0.0
    env.quaternion[:] = euler_to_quat(np.zeros(1), np.zeros(1), np.asarray([np.pi / 2], dtype=np.float32))

    obs = env.observation()

    assert abs(obs[0, 16]) < 1e-5
    assert abs(obs[0, 17] - 1.0) < 1e-5


def test_position_yaw_observation_keeps_reset_target_yaw() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=37, task="position_yaw")
    env.position[:] = np.asarray([[0.75, 0.0, 0.65]], dtype=np.float32)
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.65]], dtype=np.float32)
    env.target_yaw[:] = 0.0
    env.quaternion[:] = euler_to_quat(np.zeros(1), np.zeros(1), np.asarray([np.pi / 2], dtype=np.float32))

    obs = env.observation()

    assert abs(obs[0, 16] + 1.0) < 1e-5
    assert abs(obs[0, 17]) < 1e-5


def test_multitask_observation_uses_each_episode_task_yaw_reference() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=39, task="position_yaw")
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
    env.set_native_context(
        task_indices=np.asarray([0, 1]),
        tasks=("position_yaw", "circle"),
    )

    obs = env.observation()

    assert abs(obs[0, 16] + 1.0) < 1e-5
    assert abs(obs[1, 16]) < 1e-5
    assert abs(obs[1, 17] - 1.0) < 1e-5


def test_native_circle_step_recomputes_tangent_yaw_observation() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=41, task="circle", use_native_step=True)
    env.position[:] = np.asarray([[0.75, 0.0, 0.65]], dtype=np.float32)
    env.velocity[:] = 0.0
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.65]], dtype=np.float32)
    env.target_yaw[:] = 0.0
    env.quaternion[:] = euler_to_quat(np.zeros(1), np.zeros(1), np.asarray([np.pi / 2], dtype=np.float32))
    env.body_rates[:] = 0.0
    env._update_ranges()

    obs, _rewards, _terminals, _truncations, _info = env.step(np.zeros((1, 4), dtype=np.float32))

    assert abs(obs[0, 16]) < 1e-5
    assert abs(obs[0, 17] - 1.0) < 1e-5


def test_task_conditioned_yaw_reward_uses_circle_reference() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=29)
    env.position[:] = np.asarray([[0.75, 0.0, 0.65]], dtype=np.float32)
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.65]], dtype=np.float32)
    env.target_yaw[:] = 0.0
    env.quaternion[:] = euler_to_quat(np.zeros(1), np.zeros(1), np.asarray([np.pi / 2], dtype=np.float32))
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    actions = np.zeros((1, 4), dtype=np.float32)
    done = np.zeros(1, dtype=bool)

    position_reward = rollout_reward(env, np.zeros(1, dtype=np.float32), done, previous_error, actions, "progress_yaw_clearance")
    circle_reward = rollout_reward(
        env,
        np.zeros(1, dtype=np.float32),
        done,
        previous_error,
        actions,
        "progress_yaw_clearance",
        tasks=("circle",),
        task_indices=np.zeros(1, dtype=np.int64),
    )

    assert circle_reward[0] > position_reward[0] + 0.9
