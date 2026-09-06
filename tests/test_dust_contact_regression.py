"""Regressions for suspended wall grains and rotor-plane trapping."""

import numpy as np
from flightrl.environment import EnvironmentProfile
from flightrl.environment.airflow import Airflow
from flightrl.environment.particles import DustParticles

ROOM = np.array([-4, 8, -3, 3, 0, 3.4, 8], np.float32)


def test_wall_contact_keeps_grain_falling():
    wall = np.array([[0, 0.1, -1, 1, 0, 3]], np.float32)
    profile = EnvironmentProfile("wall", particle_count=1, grain_diameter_um=(60, 60))
    dust = DustParticles(profile, ROOM, wall, np.random.default_rng(4))
    dust.position[:] = [-0.01, 0, 1]
    for _ in range(100):
        dust.step(np.array([[1.0, 0, 0]]), 0.005)
    assert dust.active[0]
    assert dust.position[0, 0] < 0
    assert dust.position[0, 2] < 0.95
    assert dust.deposited == 0


def test_rotor_plane_has_continuous_downward_flow():
    profile = EnvironmentProfile("wake", wind_m_s=(0, 0, 0), turbulence_m_s=0)
    air = Airflow(profile, ROOM, np.empty((0, 6)), np.random.default_rng(1))
    p = np.array([0.0, 0.0, 0.65])
    points = np.array([[0.049, 0.049, 0.6499], [0.049, 0.049, 0.6501]])
    flow = air.sample(points, p, np.array([1.0, 0, 0, 0]), 1)
    assert np.all(flow[:, 2] < -0.3)
    assert abs(flow[0, 2] - flow[1, 2]) < 0.01
