from __future__ import annotations

import numpy as np

from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.env import euler_to_quat
from flightrl.sixdof.puffer_ppo import PUFFER_REWARD_MODES, puffer_rollout_reward


def test_drift_recovery_reward_penalizes_open_space_velocity_and_position_error() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=17, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 1.2
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]], dtype=np.float32)
    env.position[:] = np.asarray([[0.9, 0.0, 0.5], [0.05, 0.0, 0.5]], dtype=np.float32)
    env.velocity[:] = np.asarray([[1.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float32)
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    actions = np.zeros((2, 4), dtype=np.float32)
    done = np.zeros(2, dtype=bool)

    reward = puffer_rollout_reward(env, np.zeros(2, dtype=np.float32), done, previous_error, actions, "puffer_drift_recovery", (env.task,), np.zeros(2, dtype=np.int64))

    assert "puffer_drift_recovery" in PUFFER_REWARD_MODES
    assert reward[0] < reward[1] - 2.0


def test_aggressive_drift_reward_penalizes_stress_speed_more() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=18, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 1.2
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]], dtype=np.float32)
    env.position[:] = np.asarray([[0.4, 0.0, 0.5], [0.4, 0.0, 0.5]], dtype=np.float32)
    env.velocity[:] = np.asarray([[2.2, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32)
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    actions = np.zeros((2, 4), dtype=np.float32)
    done = np.zeros(2, dtype=bool)

    normal = puffer_rollout_reward(env, np.zeros(2, dtype=np.float32), done, previous_error, actions, "puffer_drift_recovery", (env.task,), np.zeros(2, dtype=np.int64))
    aggressive = puffer_rollout_reward(env, np.zeros(2, dtype=np.float32), done, previous_error, actions, "puffer_drift_recovery_aggressive", (env.task,), np.zeros(2, dtype=np.int64))

    assert "puffer_drift_recovery_aggressive" in PUFFER_REWARD_MODES
    assert aggressive[0] - aggressive[1] < normal[0] - normal[1] - 3.0


def test_aggressive_drift_reward_discounts_tilt_when_drifting() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=19, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 1.2
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32)
    env.position[:] = np.asarray([[0.6, 0.0, 0.5]], dtype=np.float32)
    env.velocity[:] = np.asarray([[1.4, 0.0, 0.0]], dtype=np.float32)
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    done = np.zeros(1, dtype=bool)
    zero = np.zeros((1, 4), dtype=np.float32)
    tilt = np.asarray([[0.0, 0.7, 0.7, 0.0]], dtype=np.float32)

    normal_zero = puffer_rollout_reward(env, np.zeros(1, dtype=np.float32), done, previous_error, zero, "puffer_drift_recovery", (env.task,), np.zeros(1, dtype=np.int64))
    normal_tilt = puffer_rollout_reward(env, np.zeros(1, dtype=np.float32), done, previous_error, tilt, "puffer_drift_recovery", (env.task,), np.zeros(1, dtype=np.int64))
    aggressive_zero = puffer_rollout_reward(env, np.zeros(1, dtype=np.float32), done, previous_error, zero, "puffer_drift_recovery_aggressive", (env.task,), np.zeros(1, dtype=np.int64))
    aggressive_tilt = puffer_rollout_reward(env, np.zeros(1, dtype=np.float32), done, previous_error, tilt, "puffer_drift_recovery_aggressive", (env.task,), np.zeros(1, dtype=np.int64))

    assert float((aggressive_zero - aggressive_tilt)[0]) < float((normal_zero - normal_tilt)[0])


def test_precontact_transfer_reward_penalizes_open_space_speed_and_tilt() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=20, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 1.2
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]], dtype=np.float32)
    env.position[:] = np.asarray([[0.35, 0.0, 0.5], [0.35, 0.0, 0.5]], dtype=np.float32)
    env.velocity[:] = np.asarray([[1.8, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    env.quaternion[:] = euler_to_quat(
        np.asarray([0.65, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    actions = np.zeros((2, 4), dtype=np.float32)
    done = np.zeros(2, dtype=bool)

    reward = puffer_rollout_reward(env, np.zeros(2, dtype=np.float32), done, previous_error, actions, "puffer_precontact_transfer", (env.task,), np.zeros(2, dtype=np.int64))

    assert "puffer_precontact_transfer" in PUFFER_REWARD_MODES
    assert reward[0] < reward[1] - 4.0


def test_precontact_transfer_reward_penalizes_open_space_action_jumps() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=22, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 1.2
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]], dtype=np.float32)
    env.position[:] = np.asarray([[0.05, 0.0, 0.5], [0.05, 0.0, 0.5]], dtype=np.float32)
    env.velocity[:] = 0.0
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    actions = np.asarray([[0.0, 0.7, 0.7, 0.0], [0.0, 0.7, 0.7, 0.0]], dtype=np.float32)
    previous_action = np.asarray([[0.0, 0.7, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    done = np.zeros(2, dtype=bool)

    reward = puffer_rollout_reward(
        env,
        np.zeros(2, dtype=np.float32),
        done,
        previous_error,
        actions,
        "puffer_precontact_transfer",
        (env.task,),
        np.zeros(2, dtype=np.int64),
        previous_action=previous_action,
    )

    assert reward[1] < reward[0] - 0.2


def test_precontact_transfer_reward_does_not_treat_close_obstacle_as_open_space() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=21, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 0.30
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]], dtype=np.float32)
    env.position[:] = np.asarray([[0.35, 0.0, 0.5], [0.35, 0.0, 0.5]], dtype=np.float32)
    env.velocity[:] = np.asarray([[1.8, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    previous_error = np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)
    actions = np.zeros((2, 4), dtype=np.float32)
    done = np.zeros(2, dtype=bool)

    reward = puffer_rollout_reward(env, np.zeros(2, dtype=np.float32), done, previous_error, actions, "puffer_precontact_transfer", (env.task,), np.zeros(2, dtype=np.int64))

    assert abs(float(reward[0] - reward[1])) < 0.1
