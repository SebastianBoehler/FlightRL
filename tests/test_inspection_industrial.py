"""Industrial sensor and disturbance contracts, independent of presentation."""

import numpy as np
from flightrl import _binding
from flightrl.inspection.conditions import PlantConditions
from flightrl.inspection.industrial import utility_plant
from flightrl.inspection.rollout import run_mission


def test_gusts_change_velocity_reproducibly():
    room = utility_plant()
    a, b = PlantConditions(3, room), PlantConditions(3, room)
    va, vb = np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32)
    for _ in range(50):
        a.step(va, 0.02, np.array([-2, -1.5, 1.5]), np.array([1, 0, 0, 0]), 1)
        b.step(vb, 0.02, np.array([-2, -1.5, 1.5]), np.array([1, 0, 0, 0]), 1)
    np.testing.assert_array_equal(va, vb)
    assert np.linalg.norm(va) > 0.001
    assert not np.array_equal(a.gust, np.zeros(3))
    assert np.all(a.particles >= a.room[:6:2])
    assert np.all(a.particles < a.room[1:6:2])


def test_dust_attenuation_grows_with_range():
    conditions = PlantConditions(1, utility_plant())
    conditions.aerosol.concentration = lambda _: np.ones(conditions.aerosol.shape)
    conditions.dust.active[:] = False
    conditions.dust.active[0] = True
    conditions.dust.position[0] = [-3, 0, 1]
    rgb = np.zeros((192, 256, 3), np.uint8)
    depth = np.ones((192, 256), np.float32)
    depth[:, 128:] = 7
    conditions.camera(rgb, depth, np.zeros(3), np.array([1, 0, 0, 0]))
    assert rgb[:, :128].mean() < rgb[:, 128:].mean()
    np.testing.assert_array_equal(depth[:, 128:], 7)


def test_policy_samples_are_recorded_sensor_downsamples():
    _, records, frames, _, samples = run_mission(
        utility_plant(), industrial=True, ticks=5, collect=True
    )
    assert frames.shape == (5, 192, 256, 3)
    for i, sample in enumerate(samples):
        expected = frames[i].reshape(48, 4, 64, 4, 3).mean(axis=(1, 3)).astype(np.uint8)
        np.testing.assert_array_equal(sample[0], expected)
    assert records[-1]["gust_m_s2"] != [0, 0, 0]
    assert (
        records[-1]["dust_airborne"] + records[-1]["dust_deposited"]
        == utility_plant().environment.particle_count
    )


def test_materials_preserve_geometry_and_range():
    scene = utility_plant()
    p = np.array([[-2, -1.5, 1.5]], np.float32)
    q = np.array([[1, 0, 0, 0]], np.float32)
    frames = [np.empty((1, 192, 256, 3), np.uint8) for _ in range(2)]
    depths = [np.empty((1, 192, 256), np.float32) for _ in range(2)]
    counts = [np.empty((1, 3, 2), np.int32) for _ in range(2)]
    for i in range(2):
        _binding.inspection_render(
            p,
            q,
            scene.scenario.arrays["terrain_bounds"],
            scene.scenario.arrays["terrain_obstacles"],
            scene.panels,
            frames[i],
            counts[i],
            depths[i],
            i,
            *scene.environment.render_buffers(),
        )
    np.testing.assert_array_equal(depths[0], depths[1])
    np.testing.assert_array_equal(counts[0], counts[1])
    assert np.isfinite(depths[1]).all() and (depths[1] > 0).all()
    assert not np.array_equal(frames[0], frames[1])


def test_scan_hold_corrects_disturbance_without_scene_access():
    from flightrl.inspection.industrial import IndustrialMission

    controller = IndustrialMission(np.array([-2.0, -1.5, 1.5]))
    q = np.array([1.0, 0, 0, 0])
    controller.command(np.array([-2.0, -1.5, 1.5]), q)
    command, _ = controller.command(np.array([-1.95, -1.55, 1.5]), q)
    assert command[0] < 0 and command[1] > 0
    assert command[3] > 0
    assert controller.waypoint_tolerance <= 0.1


def test_metal_lens_pass_spreads_visible_highlights_reproducibly():
    from flightrl.inspection.optics import CameraOptics

    a, b = CameraOptics(), CameraOptics()
    image = np.zeros((192, 256, 3), np.uint8)
    image[90:102, 122:134] = 255
    copy = image.copy()
    a.apply(image)
    b.apply(copy)
    np.testing.assert_array_equal(image, copy)
    assert image[96, 135].mean() > 1
    assert image[96, 128].mean() > image[96, 135].mean()
    assert image.dtype == np.uint8
