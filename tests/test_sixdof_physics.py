from __future__ import annotations

import numpy as np

from flightrl.puffer4_sixdof_export import build_sixdof_sections
from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.disturbance import SixDofDisturbanceProfile, configure_disturbance, resolve_disturbance_profile
from flightrl.sixdof.physics import LINEAR_DRAG, MASS, MOTOR_TAU


def test_crazyflie_training_randomization_changes_physics_rows() -> None:
    env = SixDofCrazyflieEnv(num_envs=16, seed=7, physics_profile="crazyflie_brushless", domain_randomization="crazyflie_training")
    env.reset(seed=7)

    assert env.physics_params.shape == (16, 9)
    assert np.ptp(env.physics_params[:, MASS]) > 0.0
    assert np.ptp(env.physics_params[:, LINEAR_DRAG]) > 0.0
    assert np.all(env.physics_params[:, MOTOR_TAU] > 0.0)


def test_native_matches_python_with_randomized_crazyflie_physics() -> None:
    kwargs = {"num_envs": 8, "seed": 13, "physics_profile": "crazyflie_brushless", "domain_randomization": "crazyflie_training"}
    python_env = SixDofCrazyflieEnv(use_native_step=False, **kwargs)
    native_env = SixDofCrazyflieEnv(use_native_step=True, **kwargs)
    np.testing.assert_allclose(python_env.physics_params, native_env.physics_params)

    rng = np.random.default_rng(13)
    for _ in range(8):
        actions = rng.uniform(-0.35, 0.35, size=(8, 4)).astype(np.float32)
        obs_py, rewards_py, terminals_py, truncations_py, _ = python_env.step(actions)
        obs_native, rewards_native, terminals_native, truncations_native, _ = native_env.step(actions)
        np.testing.assert_allclose(obs_native, obs_py, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(rewards_native, rewards_py, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(terminals_native, terminals_py)
        np.testing.assert_array_equal(truncations_native, truncations_py)


def test_puffer_sixdof_export_includes_realism_knobs() -> None:
    env_section = build_sixdof_sections(Puffer4ExportSettings())["env"]

    assert env_section["mass_kg"] == 0.036
    assert env_section["linear_drag"] == 0.08
    assert env_section["motor_tau_s"] == 0.035
    assert env_section["range_noise_std_m"] == 0.0
    assert env_section["action_lag_s"] == 0.0


def test_disturbance_profile_pushes_level_hover_in_open_space() -> None:
    env = SixDofCrazyflieEnv(num_envs=4, seed=11, reset_profile="obstacle_hover_live")
    env.position[:] = np.asarray([0.0, 0.0, 0.5], dtype=np.float32)
    env.velocity[:] = 0.0
    env.quaternion[:] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    env.target_position[:] = env.position
    env._update_ranges()
    configure_disturbance(env, SixDofDisturbanceProfile("test", world_accel_xy_m_s2=(0.5, 0.5)))

    for _ in range(20):
        env.step(np.zeros((env.num_envs, 4), dtype=np.float32))

    assert np.min(np.linalg.norm(env.velocity[:, :2], axis=1)) > 0.05


def test_disturbance_profile_can_load_measured_json(tmp_path) -> None:
    profile_path = tmp_path / "disturbance.json"
    profile_path.write_text(
        """
        {
          "disturbance_profile": {
            "name": "measured",
            "world_accel_xy_m_s2": [0.3, 0.4],
            "world_accel_z_m_s2": [-0.02, 0.01]
          }
        }
        """
    )

    profile = resolve_disturbance_profile(str(profile_path))

    assert profile.name == "measured"
    assert profile.world_accel_xy_m_s2 == (0.3, 0.4)
    assert profile.world_accel_z_m_s2 == (-0.02, 0.01)
