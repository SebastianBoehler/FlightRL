from __future__ import annotations

import json

import numpy as np
import pytest

from flightrl.mujoco import MuJoCoCrazyflieEnv, is_mujoco_available
from flightrl.mujoco.control import MuJoCoControlParams
from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.geometry import AxisAlignedObstacle, BoxRoom, normalize_quat
from flightrl.sixdof.motor_rpm import MotorRpmParams
from flightrl.sixdof.physics import (
    SixDofDomainRandomization,
    SixDofPhysicsProfile,
    sample_physics,
)
from flightrl.sixdof.sensor_model import SixDofSensorProfile, resolve_sensor_profile


@pytest.mark.parametrize(
    "updates",
    [
        {"mass_kg": 0.0},
        {"gravity_m_s2": float("nan")},
        {"linear_drag": -0.1},
        {"rate_tau_s": -0.1},
        {"thrust_scale": 0.0},
        {"max_rate_rad_s": (6.0, 6.0)},
        {"max_rate_rad_s": (6.0, float("inf"), 4.0)},
        {"motor_tau_s": -0.1},
    ],
)
def test_physics_profile_rejects_nonphysical_values(updates: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        SixDofPhysicsProfile(**updates)


@pytest.mark.parametrize(
    "updates",
    [
        {"motor_rpm_mass_scale": (0.0, 1.0)},
        {"linear_drag_scale": (-0.1, 1.0)},
        {"rate_tau_scale": (1.1, 1.0)},
        {"thrust_scale_scale": (1.0, float("nan"))},
        {"max_rate_scale": (1.0,)},
        {"motor_tau_s": (-0.1, 0.1)},
    ],
)
def test_domain_randomization_rejects_invalid_ranges(updates: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        SixDofDomainRandomization(**updates)


def test_sample_physics_validates_mode_and_count_even_when_randomization_is_off() -> None:
    profile = SixDofPhysicsProfile()
    disabled = SixDofDomainRandomization()
    rng = np.random.default_rng(1)

    with pytest.raises(ValueError, match="action mode"):
        sample_physics(profile, disabled, rng, 1, action_mode="typo")
    with pytest.raises(ValueError, match="count"):
        sample_physics(profile, disabled, rng, 0, action_mode="body_rate")


@pytest.mark.parametrize(
    "updates",
    [
        {"name": ""},
        {"range_observation_enabled": "false"},
        {"state_noise_std_m": float("nan")},
        {"velocity_noise_std_m_s": -0.1},
        {"range_dropout_prob": 1.01},
        {"action_lag_s": -0.1},
    ],
)
def test_sensor_profile_rejects_ambiguous_or_invalid_values(updates: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        SixDofSensorProfile(**updates)


def test_sensor_profile_json_rejects_numeric_strings_and_truthy_strings(tmp_path) -> None:
    path = tmp_path / "sensor.json"
    path.write_text(
        json.dumps(
            {
                "sensor_profile": {
                    "range_observation_enabled": "false",
                    "range_noise_std_m": "0.1",
                }
            }
        )
    )

    with pytest.raises((TypeError, ValueError)):
        resolve_sensor_profile(path)


@pytest.mark.parametrize("dt", [0.0, -0.01, float("nan"), float("inf")])
def test_sensor_action_alpha_requires_positive_finite_timestep(dt: float) -> None:
    with pytest.raises(ValueError, match="dt"):
        SixDofSensorProfile().action_alpha(dt)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AxisAlignedObstacle(x_min=1.0, x_max=1.0),
        lambda: AxisAlignedObstacle(z_max=float("nan")),
        lambda: BoxRoom(max_range_m=0.0),
        lambda: BoxRoom(
            obstacles=(AxisAlignedObstacle(x_min=1.9, x_max=2.1),)
        ),
    ],
)
def test_room_geometry_rejects_invalid_bounds(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "quaternions",
    [
        np.zeros((1, 4), dtype=np.float32),
        np.asarray([[1.0, 0.0, float("nan"), 0.0]], dtype=np.float32),
        np.ones((4,), dtype=np.float32),
    ],
)
def test_quaternion_normalization_rejects_undefined_inputs(
    quaternions: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="quaternion"):
        normalize_quat(quaternions)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_envs": 0},
        {"num_envs": 1.5},
        {"dt": 0.0},
        {"dt": float("nan")},
        {"task": "positon_yaw"},
        {"use_native_step": 1},
        {"use_native_step": True, "action_mode": "motor_rpm"},
    ],
)
def test_python_env_rejects_ambiguous_constructor_values(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        SixDofCrazyflieEnv(**kwargs)


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((2, 4), dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0, float("nan")]], dtype=np.float32),
        [["0", "0", "0", "0"]],
    ],
)
def test_python_env_rejects_malformed_action_batches(actions) -> None:
    env = SixDofCrazyflieEnv(num_envs=1)
    with pytest.raises((TypeError, ValueError)):
        env.step(actions)


@pytest.mark.parametrize(
    "done",
    [
        np.asarray([True, False]),
        np.asarray([2], dtype=np.uint8),
        np.asarray(["false"]),
    ],
)
def test_python_env_rejects_malformed_reset_masks(done: np.ndarray) -> None:
    env = SixDofCrazyflieEnv(num_envs=1)
    with pytest.raises((TypeError, ValueError)):
        env.reset_done(done)


def test_python_env_rejects_invalid_native_context_arrays() -> None:
    env = SixDofCrazyflieEnv(num_envs=2)

    with pytest.raises(ValueError, match="task indices"):
        env.set_native_context(task_indices=np.asarray([0]), tasks=("circle",))
    with pytest.raises(ValueError, match="task indices"):
        env.set_native_context(task_indices=np.asarray([0, 1]), tasks=("circle",))
    with pytest.raises(ValueError, match="previous error"):
        env.set_native_context(previous_error=np.asarray([0.0, float("nan")]))


@pytest.mark.parametrize(
    "updates",
    [
        {"hover_rpm": 0.0},
        {"max_rpm": 10_000.0, "hover_rpm": 20_000.0},
        {"motor_tau_s": -0.1},
        {"physics_substeps": 1.5},
        {"arm_length_m": float("nan")},
        {"ixx": 0.0},
        {"yaw_torque_gain": -0.1},
        {"angular_drag": -0.1},
    ],
)
def test_motor_rpm_profile_rejects_nonphysical_values(updates: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        MotorRpmParams(**updates)


@pytest.mark.parametrize(
    "updates",
    [
        {"mass_kg": float("nan")},
        {"rate_kp": -0.1},
        {"rate_tau_s": -0.1},
        {"max_rate_rad_s": (6.0, 0.0, 4.0)},
    ],
)
def test_mujoco_control_rejects_nonphysical_values(updates: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        MuJoCoControlParams(**updates)


@pytest.mark.skipif(not is_mujoco_available(), reason="MuJoCo is not installed")
def test_mujoco_env_uses_same_batch_and_constructor_validation() -> None:
    with pytest.raises(ValueError, match="task"):
        MuJoCoCrazyflieEnv(task="positon_yaw")
    env = MuJoCoCrazyflieEnv(num_envs=1)
    with pytest.raises(ValueError, match="action"):
        env.step(np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="reset mask"):
        env.reset_done(np.asarray([2], dtype=np.uint8))
