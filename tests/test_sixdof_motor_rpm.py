from __future__ import annotations

import numpy as np

from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.physics import MASS, MOTOR_TAU


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


def test_puffer_parameter_profile_matches_pinned_motor_scale() -> None:
    env = SixDofCrazyflieEnv(
        num_envs=1,
        seed=13,
        action_mode="motor_rpm",
        physics_profile="puffer_parameters",
    )

    assert env.mass == 0.027
    assert env.motor_params.max_rpm == 21702.0
    assert env.motor_params.motor_tau_s == 0.150
    assert env.motor_params.physics_substeps == 5
    np.testing.assert_allclose(env.motor_hover_rpm, 14475.809, rtol=1e-6)


def test_motor_rpm_profile_can_be_selected_independently() -> None:
    env = SixDofCrazyflieEnv(
        num_envs=1,
        seed=17,
        action_mode="motor_rpm",
        motor_rpm_profile="puffer_parameters",
    )

    assert env.mass == 0.036
    assert env.motor_params.max_rpm == 21702.0


def test_motor_rpm_uses_per_environment_randomized_mass() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=19, action_mode="motor_rpm")
    env.position[:] = (0.0, 0.0, 1.0)
    env.velocity[:] = 0.0
    env.physics_params[:, MASS] = (env.mass, 2.0 * env.mass)

    env.step(np.zeros((2, 4), dtype=np.float32))

    assert abs(float(env.velocity[0, 2])) < 1e-3
    assert float(env.velocity[1, 2]) < -0.04


def test_motor_rpm_uses_per_environment_randomized_motor_tau() -> None:
    env = SixDofCrazyflieEnv(num_envs=2, seed=23, action_mode="motor_rpm")
    env.physics_params[:, MOTOR_TAU] = (0.005, 0.5)

    env.step(np.ones((2, 4), dtype=np.float32))

    assert env.motor_rpm[0, 0] > env.motor_rpm[1, 0]
