from __future__ import annotations

from pathlib import Path

import numpy as np

from flightrl import load_config, make_env
from flightrl.observation_schema import CRAZYFLIE_TELEMETRY_BASE_DIM, RANGE_RATE_SENSOR_DIM, RANGE_SENSOR_DIM, TTC_SENSOR_DIM


ROOT = Path(__file__).resolve().parents[1]


def test_crazyflie_telemetry_observation_shape_tracks_action_dim() -> None:
    config = load_config(ROOT / "configs" / "tasks" / "crazyflie_hover.toml")

    assert config.sensors.include_crazyflie_telemetry is True
    assert config.sensors.include_range_sensor is True
    assert config.observation_dim == CRAZYFLIE_TELEMETRY_BASE_DIM + RANGE_SENSOR_DIM + config.action_dim


def test_crazyflie_telemetry_observation_is_finite_and_updates() -> None:
    config = load_config(
        ROOT / "configs" / "tasks" / "crazyflie_hover.toml",
        overrides={"environment": {"num_envs": 1}},
    )
    env = make_env(config, seed=31)
    obs, _ = env.reset(seed=31)
    initial = obs.copy()

    actions = np.array([[0.2, 0.1]], dtype=np.float32)
    next_obs, rewards, terminals, truncations, _ = env.step(actions)
    env.close()

    assert next_obs.shape == (1, config.observation_dim)
    assert np.isfinite(next_obs).all()
    assert not np.allclose(initial, next_obs)
    assert rewards.shape == (1,)
    assert terminals.shape == (1,)
    assert truncations.shape == (1,)


def test_range_sensor_can_be_enabled_without_placeholder_failure() -> None:
    config = load_config(
        ROOT / "configs" / "tasks" / "hover.toml",
        overrides={"sensors": {"include_range_sensor": True}},
    )

    assert config.observation_dim > RANGE_SENSOR_DIM


def test_ttc_range_rate_observation_shape_tracks_schema() -> None:
    config = load_config(
        ROOT / "configs" / "tasks" / "crazyflie_hover.toml",
        overrides={"sensors": {"include_range_rate_sensor": True, "include_ttc_sensor": True}},
    )

    assert config.observation_dim == CRAZYFLIE_TELEMETRY_BASE_DIM + RANGE_SENSOR_DIM + RANGE_RATE_SENSOR_DIM + TTC_SENSOR_DIM + config.action_dim


def test_native_range_rate_and_ttc_observations_are_physical() -> None:
    config = load_config(
        ROOT / "configs" / "tasks" / "hover.toml",
        overrides={
            "environment": {
                "num_envs": 1,
                "action_mode": "hover_command",
                "reset_mode": "deterministic",
            },
            "sensors": {
                "include_position": False,
                "include_velocity": False,
                "include_attitude": False,
                "include_angular_velocity": False,
                "include_target_vector": False,
                "include_previous_action": False,
                "include_health": False,
                "include_imu": False,
                "include_range_rate_sensor": True,
                "include_ttc_sensor": True,
            },
            "task": {
                "fixed_start": [7.0, 2.0],
                "fixed_target": [7.0, 2.0],
                "target_bounds": [7.0, 7.0, 2.0, 2.0],
            },
        },
    )
    env = make_env(config, seed=37)
    observations, _ = env.reset(seed=37)

    assert observations[0, 6] == np.float32(0.25)
    for _ in range(22):
        observations, *_ = env.step(
            np.asarray([[-1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        )
    snapshot = env.snapshot()
    env.close()

    assert snapshot["vx"] > 0.0
    assert observations[0, 0] < 0.0
    assert observations[0, 1] > 0.0
    assert observations[0, 7] > 0.0
