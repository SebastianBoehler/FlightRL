"""Physical coupling and authored appearance, independent of mission success."""

from dataclasses import replace
import numpy as np
import pytest
from flightrl.environment import EnvironmentProfile
from flightrl.environment.airflow import Airflow
from flightrl.environment.particles import DustParticles
from flightrl.environment.aerosol import AerosolCamera
from flightrl.inspection.industrial import utility_plant
from flightrl.inspection_scene import compile_inspection_scene
from flightrl import _binding

ROOM = np.array([-4, 8, -3, 3, 0, 3.4, 8], np.float32)
EMPTY = np.empty((0, 6), np.float32)
Q = np.array([1, 0, 0, 0], float)


def test_rotor_downwash_dominates_return_flow_and_scales_with_thrust():
    profile = EnvironmentProfile("test", wind_m_s=(0, 0, 0), turbulence_m_s=0)
    air = Airflow(profile, ROOM, EMPTY, np.random.default_rng(1))
    p = np.array([0, 0, 1.5])
    points = np.array([[0, 0, 1.2], [0, 0, 1.8], [2, 0, 1.2]])
    stopped = air.sample(points, p, Q, 0)
    wake = air.sample(points, p, Q, 1)
    assert np.allclose(stopped, 0)
    assert (
        wake[0, 2] < -0.1
        and abs(wake[1, 2]) < abs(wake[0, 2]) * 0.01
        and abs(wake[2, 2]) < 0.001
    )
    assert np.allclose(air.sample(p, p, Q, 1, wake=False), 0)
    assert air.sample(points, p, Q, 4)[0, 2] == pytest.approx(2 * wake[0, 2])


def test_downwash_resuspends_settled_floor_dust():
    profile = EnvironmentProfile(
        "test",
        wind_m_s=(0, 0, 0),
        turbulence_m_s=0,
        particle_count=1,
        resuspension_m_s=0.01,
    )
    air = Airflow(profile, ROOM, EMPTY, np.random.default_rng(2))
    dust = DustParticles(profile, ROOM, EMPTY, np.random.default_rng(3))
    dust.position[:] = [0.1, 0, 0.001]
    dust.active[:] = False
    flow = air.sample(dust.position, np.array([0, 0, 0.5]), Q, 1)
    assert flow[0, 2] > 0
    dust.step(flow, 0.02)
    assert dust.active[0] and dust.resuspended == 1


def test_swept_dust_cannot_cross_thin_partition():
    wall = np.array([[0, 0.01, -1, 1, 0, 3]], np.float32)
    profile = EnvironmentProfile("test", particle_count=1)
    dust = DustParticles(profile, ROOM, wall, np.random.default_rng(4))
    dust.position[:] = [-0.1, 0, 1]
    dust.velocity[:] = [20, 0, 0]
    dust.step(np.array([[20.0, 0, 0]]), 0.02)
    assert dust.position[0, 0] < 0 and dust.active[0]
    assert dust.deposited == 0


def test_settling_deposits_and_concentration_tracks_airborne_mass():
    profile = EnvironmentProfile("test", particle_count=64)
    dust = DustParticles(profile, ROOM, EMPTY, np.random.default_rng(5))
    camera = AerosolCamera(profile, ROOM)
    mass = camera.concentration(dust).sum()
    dust.active[:32] = False
    assert camera.concentration(dust).sum() == pytest.approx(mass / 2)
    dust.position[32:, 2] = 0.001
    dust.velocity[32:, 2] = -1
    for _ in range(25):
        dust.step(np.zeros((64, 3)), 0.02)
    assert not dust.active.any()
    assert camera.concentration(dust).sum() == 0


def render(profile):
    scene = utility_plant()
    rgb = np.empty((1, 192, 256, 3), np.uint8)
    depth = np.empty((1, 192, 256), np.float32)
    _binding.inspection_render(
        np.array([[-2, -1.5, 1.5]], np.float32),
        np.array([[0.70710678, 0, 0, 0.70710678]], np.float32),
        ROOM,
        scene.scenario.arrays["terrain_obstacles"],
        scene.panels,
        rgb,
        np.zeros((1, 3, 2), np.int32),
        depth,
        1,
        *profile.render_buffers(),
    )
    return rgb, depth


def test_scene_lighting_and_material_changes_preserve_geometry():
    base = utility_plant().environment
    dark, depth = render(replace(base, ambient=0.05, lights=()))
    lit, other = render(base)
    assert lit.mean() > dark.mean()
    np.testing.assert_array_equal(depth, other)
    daylight, _ = render(
        replace(base, windows=((-3.8, -0.2, 2.99, 3.01, 2.15, 3.2),), sun_strength=1.8)
    )
    assert not np.array_equal(daylight, lit)
    assert base.windows == () and base.sun_strength == 0
    tinted, _ = render(replace(base, equipment_rgb=(140, 65, 40)))
    assert not np.array_equal(tinted, lit)


def test_environment_is_immutable_and_bound_to_scene_identity():
    scene = utility_plant()
    profile = replace(scene.environment, wind_m_s=[1, 0, 0])
    assert isinstance(profile.wind_m_s, tuple)
    changed = compile_inspection_scene(
        scene.scenario, scene.panels, scene.evaluator_ids, environment=profile
    )
    assert changed.manifest["sha256"] != scene.manifest["sha256"]


@pytest.mark.parametrize(
    "change",
    [
        {"dust_extinction_per_m": -1},
        {"wind_m_s": (float("nan"), 0, 0)},
        {"particle_count": 0},
        {"correlation_s": 0},
        {"equipment_roughness": 2},
        {"lights": ((1, 2, 3, -1, 1, 1, 1),)},
    ],
)
def test_invalid_environment_rejected(change):
    with pytest.raises(ValueError):
        EnvironmentProfile("invalid", **change)


def test_dust_loading_changes_visibility_not_arbitrary_drone_force():
    from flightrl.environment.simulation import EnvironmentSimulation

    scene = utility_plant()
    heavy = compile_inspection_scene(
        scene.scenario,
        scene.panels,
        scene.evaluator_ids,
        environment=replace(scene.environment, dust_extinction_per_m=0.8),
    )
    normal_sim = EnvironmentSimulation(9, scene)
    heavy_sim = EnvironmentSimulation(9, heavy)
    p = np.array([-2.0, -1.5, 1.5])
    v = np.zeros((1, 3))
    w = v.copy()
    for _ in range(3):
        normal_sim.step(v, 0.02, p, Q, 1)
        heavy_sim.step(w, 0.02, p, Q, 1)
    np.testing.assert_array_equal(v, w)
    a = np.full((192, 256, 3), 200, np.uint8)
    b = a.copy()
    depth = np.full((192, 256), 3, np.float32)
    normal_sim.camera(a, depth, p, Q)
    heavy_sim.camera(b, depth, p, Q)
    assert heavy_sim.aerosol.mean_transmission < normal_sim.aerosol.mean_transmission
    assert not np.array_equal(a, b)


def test_aerosol_lighting_is_shadowed_by_equipment():
    from flightrl.environment.lighting import volume_lighting

    profile = EnvironmentProfile("light", ambient=0.1, lights=((2, 0, 1, 1, 1, 1, 5),))
    points = np.array([[-1.0, 0, 1]])
    clear = volume_lighting(profile, points, EMPTY, ROOM)
    shadow = volume_lighting(profile, points, np.array([[0, 0.2, -1, 1, 0, 2]]), ROOM)
    assert np.all(clear > shadow)


def test_settled_bed_forms_rotor_plume_without_creating_particles():
    profile = EnvironmentProfile(
        "bed",
        wind_m_s=(0, 0, 0),
        turbulence_m_s=0,
        particle_count=4096,
        settled_fraction=1,
    )
    dust = DustParticles(profile, ROOM, EMPTY, np.random.default_rng(2))
    air = Airflow(profile, ROOM, EMPTY, np.random.default_rng(3))
    drone = np.array([0, 0, 0.7])
    assert not dust.active.any()
    for _ in range(50):
        air.advance(0.02)
        dust.step(air.sample(dust.position, drone, Q, 0), 0.02)
    assert not dust.active.any()
    for _ in range(300):
        air.advance(0.02)
        dust.step(air.sample(dust.position, drone, Q, 1), 0.02)
    assert dust.active.sum() > 10
    assert dust.position[dust.active, 2].max() > 0.15
    assert len(dust.position) == profile.particle_count
    assert np.isfinite(dust.position).all()


def test_gravity_and_drag_settle_grains_at_physical_terminal_speed():
    profile = EnvironmentProfile("grain", grain_diameter_um=(20, 20), particle_count=1)
    dust = DustParticles(profile, ROOM, EMPTY, np.random.default_rng(3))
    dust.position[:] = [0, 0, 2]
    start = dust.position[0, 2]
    for _ in range(50):
        dust.step(np.zeros((1, 3)), 0.02)
    stokes_speed = dust.gravity * dust.relaxation_s[0]
    assert -dust.velocity[0, 2] == pytest.approx(stokes_speed, rel=0.02)
    assert start - dust.position[0, 2] == pytest.approx(stokes_speed, rel=0.02)


def test_ground_return_has_no_discontinuity_at_drone_height():
    profile = EnvironmentProfile("continuity", wind_m_s=(0, 0, 0), turbulence_m_s=0)
    air = Airflow(profile, ROOM, EMPTY, np.random.default_rng(1))
    points = np.array([[0.7, 0, 1.4999], [0.7, 0, 1.5001]])
    flow = air.sample(points, np.array([0, 0, 1.5]), Q, 1)
    assert np.linalg.norm(flow[0] - flow[1]) < 0.001


def test_local_bed_starts_on_ground_and_stays_there_without_airflow():
    profile = EnvironmentProfile(
        "corner", settled_fraction=1, dust_bed_bounds=(-3.8, -2.65, -2.9, -1.85)
    )
    dust = DustParticles(profile, ROOM, EMPTY, np.random.default_rng(1))
    before = dust.position.copy()
    for _ in range(50):
        dust.step(np.zeros_like(before), 0.02)
    np.testing.assert_array_equal(dust.position, before)
    assert not dust.active.any()
    assert np.all(dust.position[:, 0] < -2.65)
    assert np.all(dust.position[:, 1] < -1.85)


def test_room_dust_and_diagnostic_grid_do_not_follow_drone_pose():
    from flightrl.environment.simulation import EnvironmentSimulation

    base = utility_plant()
    profile = replace(
        base.environment, settled_fraction=1, wind_m_s=(0, 0, 0), turbulence_m_s=0
    )
    scene = compile_inspection_scene(
        base.scenario, base.panels, base.evaluator_ids, environment=profile
    )
    sim = EnvironmentSimulation(3, scene)
    before = sim.dust.position.copy()
    grid = sim.flow_points.copy()
    for pose in ([-2, -1.5, 1.5], [6, 0, 2]):
        sim.step(np.zeros((1, 3)), 0.02, np.array(pose), Q, 0)
        np.testing.assert_array_equal(sim.dust.position, before)
        np.testing.assert_array_equal(sim.flow_samples[:, :3], grid)
    record = sim.record()
    assert not record["particles"]
    assert len(record["settled_particles"]) == profile.particle_count
