from __future__ import annotations

import numpy as np

from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.env import euler_to_quat
from flightrl.sixdof.puffer_ppo import PUFFER_REWARD_MODES, puffer_rollout_reward


def test_precontact_drift_brake_reward_prefers_body_frame_braking_action() -> None:
    env = SixDofCrazyflieEnv(num_envs=3, seed=31, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 1.2
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.5]] * 3, dtype=np.float32)
    env.position[:] = np.asarray([[0.2, 0.0, 0.5]] * 3, dtype=np.float32)
    env.velocity[:] = np.asarray([[1.4, 0.0, 0.0]] * 3, dtype=np.float32)
    env.quaternion[:] = euler_to_quat(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    actions = np.asarray(
        [
            [0.0, 0.0, -0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
        ],
        dtype=np.float32,
    )

    reward = puffer_rollout_reward(
        env,
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=bool),
        previous_error,
        actions,
        "puffer_precontact_drift_brake",
        (env.task,),
        np.zeros(3, dtype=np.int64),
    )

    assert "puffer_precontact_drift_brake" in PUFFER_REWARD_MODES
    assert reward[0] > reward[1] + 0.2
    assert reward[2] < reward[1] - 0.2


def test_precontact_drift_brake_reward_ignores_brake_alignment_when_close() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=32, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 0.32
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.5]] * 2, dtype=np.float32)
    env.position[:] = np.asarray([[0.2, 0.0, 0.5]] * 2, dtype=np.float32)
    env.velocity[:] = np.asarray([[1.4, 0.0, 0.0]] * 2, dtype=np.float32)
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    actions = np.asarray([[0.0, 0.0, -0.5, 0.0], [0.0, 0.0, 0.5, 0.0]], dtype=np.float32)

    reward = puffer_rollout_reward(
        env,
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=bool),
        previous_error,
        actions,
        "puffer_precontact_drift_brake",
        (env.task,),
        np.zeros(2, dtype=np.int64),
    )

    assert abs(float(reward[0] - reward[1])) < 0.05


def test_startup_drift_recovery_reward_prefers_speed_reduction() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=33, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 1.2
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.5]] * 2, dtype=np.float32)
    env.position[:] = np.asarray([[0.2, 0.0, 0.5]] * 2, dtype=np.float32)
    env.velocity[:] = np.asarray([[0.8, 0.0, 0.0]] * 2, dtype=np.float32)
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)

    reward = puffer_rollout_reward(
        env,
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=bool),
        previous_error,
        np.zeros((2, 4), dtype=np.float32),
        "puffer_startup_drift_recovery",
        (env.task,),
        np.zeros(2, dtype=np.int64),
        previous_horizontal_speed=np.asarray([1.4, 0.4], dtype=np.float32),
    )

    assert "puffer_startup_drift_recovery" in PUFFER_REWARD_MODES
    assert reward[0] > reward[1] + 2.0


def test_startup_drift_recovery_reward_prefers_brake_direction() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=34, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 1.2
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.5]] * 2, dtype=np.float32)
    env.position[:] = np.asarray([[0.2, 0.0, 0.5]] * 2, dtype=np.float32)
    env.velocity[:] = np.asarray([[1.2, 0.0, 0.0]] * 2, dtype=np.float32)
    env.quaternion[:] = euler_to_quat(np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    actions = np.asarray([[0.0, 0.0, -0.5, 0.0], [0.0, 0.0, 0.5, 0.0]], dtype=np.float32)

    reward = puffer_rollout_reward(
        env,
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=bool),
        previous_error,
        actions,
        "puffer_startup_drift_recovery",
        (env.task,),
        np.zeros(2, dtype=np.int64),
        previous_horizontal_speed=np.asarray([1.4, 1.4], dtype=np.float32),
    )

    assert reward[0] > reward[1] + 0.7
