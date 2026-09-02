from __future__ import annotations

import numpy as np

from flightrl.sixdof.curriculum import RESET_PROFILES, sample_reset
from flightrl.sixdof.env import euler_to_quat
from flightrl.sixdof.geometry import BoxRoom
from flightrl.sixdof.yaw import circle_tangent_yaw
from flightrl.sixdof import SixDofEnv


def test_circle_profiles_are_registered() -> None:
    assert {"circle_easy", "circle_recovery"} <= set(RESET_PROFILES)


def test_circle_profile_samples_target_near_orbit_radius() -> None:
    profile = RESET_PROFILES["circle_easy"]
    position, _roll, _pitch, _yaw, target, _target_yaw = sample_reset(profile, np.random.default_rng(7), 128, BoxRoom())
    radius = np.linalg.norm((position - target)[:, :2], axis=1)

    assert float(np.median(radius)) > 0.55
    assert float(np.median(radius)) < 0.95
    assert np.all(target[:, 2] >= profile.target_z_range[0])
    assert np.all(target[:, 2] <= profile.target_z_range[1])


def test_circle_profile_initial_yaw_is_aligned_to_tangent() -> None:
    profile = RESET_PROFILES["circle_recovery"]
    position, roll, pitch, yaw, target, _target_yaw = sample_reset(profile, np.random.default_rng(11), 256, BoxRoom())
    env = SixDofEnv(num_envs=256, seed=11, task="circle", reset_profile="circle_recovery")
    env.position[:] = position
    env.target_position[:] = target
    env.quaternion[:] = euler_to_quat(roll, pitch, yaw)
    yaw_error = np.abs(((circle_tangent_yaw(env) - yaw + np.pi) % (2.0 * np.pi)) - np.pi)

    assert float(np.max(yaw_error)) <= profile.target_yaw_offset_abs + 1e-5
