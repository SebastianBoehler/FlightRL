from __future__ import annotations

import numpy as np

from flightrl.sixdof import SixDofEnv
from flightrl.sixdof.policies import teacher_actions


def test_hover_profile_starts_without_velocity() -> None:
    env = SixDofEnv(num_envs=16, seed=31, task="obstacle_avoidance", reset_profile="obstacle_hover_live")

    assert np.max(np.linalg.norm(env.velocity, axis=1)) == 0.0


def test_drift_recovery_profile_samples_initial_horizontal_velocity() -> None:
    env = SixDofEnv(num_envs=128, seed=32, task="obstacle_avoidance", reset_profile="obstacle_hover_drift_recovery")
    horizontal_speed = np.linalg.norm(env.velocity[:, :2], axis=1)

    assert np.quantile(horizontal_speed, 0.75) > 0.45
    assert np.max(np.abs(env.velocity[:, 2])) > 0.02


def test_open_drift_stress_profile_starts_away_from_walls() -> None:
    env = SixDofEnv(num_envs=128, seed=33, task="obstacle_avoidance", reset_profile="obstacle_hover_open_drift_stress")
    horizontal_clearance = np.min(env.ranges_m[:, :4], axis=1)

    assert np.min(horizontal_clearance) > 1.4
    assert np.max(np.linalg.norm(env.velocity, axis=1)) == 0.0


def test_raw_transfer_stress_profile_samples_failed_precontact_speed_tail() -> None:
    env = SixDofEnv(num_envs=512, seed=36, task="obstacle_avoidance", reset_profile="obstacle_hover_raw_transfer_stress")
    horizontal_speed = np.linalg.norm(env.velocity[:, :2], axis=1)
    horizontal_clearance = np.min(env.ranges_m[:, :4], axis=1)

    assert np.min(horizontal_clearance) > 1.4
    assert np.quantile(horizontal_speed, 0.95) > 2.0
    assert np.max(np.abs(env.velocity[:, 2])) > 0.05


def test_aggressive_open_stress_teacher_adds_recovery_authority() -> None:
    env = SixDofEnv(num_envs=1, seed=34, task="obstacle_avoidance", reset_profile="obstacle_hover_open_drift_stress")
    env.target_position[:] = env.position
    env.velocity[:] = np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32)

    default = teacher_actions(env, task="obstacle_avoidance")[0]
    env.teacher_profile = "aggressive_open_stress"
    aggressive = teacher_actions(env, task="obstacle_avoidance")[0]

    assert abs(aggressive[2]) > abs(default[2]) * 3.0


def test_open_space_stress_teacher_keeps_close_obstacle_default() -> None:
    env = SixDofEnv(num_envs=1, seed=35, task="obstacle_avoidance", reset_profile="obstacle_hover_open_drift_stress")
    env.target_position[:] = env.position
    env.velocity[:] = np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32)
    default = teacher_actions(env, task="obstacle_avoidance")[0]
    env.teacher_profile = "open_space_stress"
    open_action = teacher_actions(env, task="obstacle_avoidance")[0]
    env.ranges_m[:, :4] = np.asarray([[0.2, 0.25, 0.3, 0.35]], dtype=np.float32)
    env.teacher_profile = "default"
    default_close = teacher_actions(env, task="obstacle_avoidance")[0]
    env.teacher_profile = "open_space_stress"
    close_action = teacher_actions(env, task="obstacle_avoidance")[0]

    assert abs(open_action[2]) > abs(default[2]) * 2.0
    np.testing.assert_allclose(close_action, default_close, atol=1e-6)


def test_bounded_recovery_teacher_stays_inside_live_action_envelope() -> None:
    env = SixDofEnv(num_envs=1, seed=37, task="obstacle_avoidance", reset_profile="obstacle_hover_open_drift_stress")
    env.target_position[:] = env.position
    env.velocity[:] = np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32)
    default = teacher_actions(env, task="obstacle_avoidance")[0]
    env.teacher_profile = "aggressive_open_stress"
    aggressive = teacher_actions(env, task="obstacle_avoidance")[0]
    env.teacher_profile = "bounded_recovery"
    bounded = teacher_actions(env, task="obstacle_avoidance")[0]

    assert abs(default[2]) < abs(bounded[2]) < abs(aggressive[2])
    assert np.max(np.abs(bounded[[1, 2]])) <= 0.64
    assert -0.25 <= bounded[0] <= 0.35


def test_bounded_recovery_teacher_keeps_close_obstacle_default() -> None:
    env = SixDofEnv(num_envs=1, seed=38, task="obstacle_avoidance", reset_profile="obstacle_hover_open_drift_stress")
    env.target_position[:] = env.position
    env.velocity[:] = np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32)
    env.ranges_m[:, :4] = np.asarray([[0.2, 0.25, 0.3, 0.35]], dtype=np.float32)
    default = teacher_actions(env, task="obstacle_avoidance")[0]
    env.teacher_profile = "bounded_recovery"
    bounded = teacher_actions(env, task="obstacle_avoidance")[0]

    np.testing.assert_allclose(bounded, default, atol=1e-6)
