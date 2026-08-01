from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from flightrl.mujoco import MuJoCoCrazyflieEnv, is_mujoco_available
from flightrl.mujoco.control import MuJoCoControlParams, rate_control_torque
from flightrl.mujoco.model import build_crazyflie_mjcf
from flightrl.sixdof import SixDofCrazyflieEnv


def test_brushless_model_uses_official_motor_span_and_propeller_diameter() -> None:
    root = ET.fromstring(build_crazyflie_mjcf())
    front_left = _geom_vector(root, "rotor_front_left", "pos")
    rear_right = _geom_vector(root, "rotor_rear_right", "pos")
    front_right = _geom_vector(root, "rotor_front_right", "pos")
    rear_left = _geom_vector(root, "rotor_rear_left", "pos")

    assert np.linalg.norm(front_left - rear_right) == pytest.approx(0.100)
    assert np.linalg.norm(front_right - rear_left) == pytest.approx(0.100)
    assert front_left[0] > 0.0 and front_left[1] > 0.0
    assert front_right[0] > 0.0 and front_right[1] < 0.0
    for name in _ROTOR_NAMES:
        radius = _geom_vector(root, name, "size")[0]
        assert 2.0 * radius == pytest.approx(0.055)


def test_aideck_camera_is_outside_propellers_and_forward_axis_is_clear() -> None:
    root = ET.fromstring(build_crazyflie_mjcf())
    camera = _element_vector(root, "camera", "aideck", "pos")

    for name in _ROTOR_NAMES:
        rotor = _geom_vector(root, name, "pos")
        rotor_radius, rotor_half_height = _geom_vector(root, name, "size")
        radial_distance = np.linalg.norm(camera[:2] - rotor[:2])
        vertical_distance = abs(camera[2] - rotor[2])
        assert radial_distance > rotor_radius or vertical_distance > rotor_half_height
        assert abs(rotor[1]) > rotor_radius


def test_rate_controller_has_zero_torque_at_commanded_rate() -> None:
    control = MuJoCoControlParams()
    command = np.asarray((2.1, -1.3, 0.7), dtype=np.float64)

    torque = rate_control_torque(command, command, control)

    np.testing.assert_allclose(torque, 0.0, atol=1.0e-15)


def test_mujoco_reset_matches_python_initial_velocity_profile() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    kwargs = {
        "num_envs": 32,
        "seed": 731,
        "task": "obstacle_avoidance",
        "reset_profile": "obstacle_hover_drift_recovery",
    }

    python_env = SixDofCrazyflieEnv(**kwargs)
    mujoco_env = MuJoCoCrazyflieEnv(**kwargs)

    np.testing.assert_array_equal(mujoco_env.velocity, python_env.velocity)
    assert np.max(np.linalg.norm(mujoco_env.velocity[:, :2], axis=1)) > 0.0


def test_mujoco_propeller_contact_terminates_inside_center_bounds() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    env = MuJoCoCrazyflieEnv(num_envs=1, seed=17)
    data = env.data[0]
    data.qpos[:3] = (env.room.x_max - 0.06, 0.0, 1.0)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[:] = 0.0
    env.mujoco.mj_forward(env.model, data)
    assert env.room.contains(np.asarray((data.qpos[:3],)))[0]

    _obs, _rewards, terminals, _truncations, _infos = env.step(
        np.zeros((1, 4), dtype=np.float32)
    )

    assert env.forbidden_contact_counts[0] > 0
    assert terminals[0] == 1


def _geom_vector(root: ET.Element, name: str, attribute: str) -> np.ndarray:
    return _element_vector(root, "geom", name, attribute)


def _element_vector(
    root: ET.Element,
    tag: str,
    name: str,
    attribute: str,
) -> np.ndarray:
    element = root.find(f".//{tag}[@name='{name}']")
    assert element is not None
    return np.fromstring(element.attrib[attribute], sep=" ")


_ROTOR_NAMES = (
    "rotor_front_left",
    "rotor_rear_right",
    "rotor_front_right",
    "rotor_rear_left",
)
