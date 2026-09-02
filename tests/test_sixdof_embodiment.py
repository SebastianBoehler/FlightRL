from __future__ import annotations

import numpy as np

from flightrl.sixdof import SixDofEnv
from flightrl.sixdof.embodiment import (
    EMBODIMENT_FIELDS,
    EmbodimentDescriptor,
)
from flightrl.sixdof.physics import SixDofPhysicsProfile


def test_embodiment_descriptor_has_stable_physical_order() -> None:
    profile = SixDofPhysicsProfile(
        mass_kg=0.5,
        linear_drag=0.2,
        rate_tau_s=0.03,
        thrust_scale=1.25,
        max_rate_rad_s=(5.0, 6.0, 7.0),
        motor_tau_s=0.04,
    )

    descriptor = EmbodimentDescriptor.from_physics_profile(profile)

    assert EMBODIMENT_FIELDS == (
        "mass_kg",
        "linear_drag",
        "rate_tau_s",
        "thrust_scale",
        "max_rate_roll_rad_s",
        "max_rate_pitch_rad_s",
        "max_rate_yaw_rad_s",
        "motor_tau_s",
    )
    np.testing.assert_array_equal(
        descriptor.as_array(),
        np.asarray([0.5, 0.2, 0.03, 1.25, 5.0, 6.0, 7.0, 0.04], dtype=np.float32),
    )


def test_environment_exposes_each_sampled_embodiment() -> None:
    env = SixDofEnv(
        num_envs=16,
        seed=9,
        physics_profile="crazyflie_brushless",
        domain_randomization="crazyflie_training",
    )

    descriptors = env.embodiment_descriptors()

    assert descriptors.shape == (16, len(EMBODIMENT_FIELDS))
    np.testing.assert_array_equal(descriptors[:, 0], env.physics_params[:, 0])
    np.testing.assert_array_equal(descriptors[:, 1:], env.physics_params[:, 2:])
