from __future__ import annotations

import numpy as np

from flightrl.sixdof import SixDofCrazyflieEnv


def test_motor_rpm_mode_initializes_hover_rpm_state() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=3, action_mode="motor_rpm")

    assert env.motor_rpm.shape == (4, 4)
    assert env.motor_hover_rpm > 0.0
    np.testing.assert_allclose(env.motor_rpm, env.motor_hover_rpm)


def test_motor_rpm_zero_action_holds_level_altitude_short_term() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=5, action_mode="motor_rpm")
    start_z = env.position[:, 2].copy()

    for _ in range(20):
        env.step(np.zeros((env.num_envs, 4), dtype=np.float32))

    np.testing.assert_allclose(env.position[:, 2], start_z, atol=0.02)


def test_motor_rpm_asymmetric_action_changes_body_rates() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=7, action_mode="motor_rpm")
    action = np.asarray([[0.4, -0.4, 0.4, -0.4]], dtype=np.float32)

    env.step(action)

    assert np.linalg.norm(env.body_rates[0]) > 0.0
    assert not np.allclose(env.motor_rpm[0], env.motor_hover_rpm)


def test_motor_rpm_mode_is_explicitly_sim_only() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=11, action_mode="motor_rpm")

    assert env.hardware_action_interface == "sim_only_motor_rpm"
