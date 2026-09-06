import base64
import numpy as np
import pytest
from flightrl.realism.physics import ContactWorld
from flightrl.realism.scene import decode_scene


def scene():
    vertices = np.array(
        [
            [-10, -10, 0],
            [10, -10, 0],
            [10, 10, 0],
            [-10, 10, 0],
            [2, -5, 0],
            [2, -5, 5],
            [2, 5, 5],
            [2, 5, 0],
        ],
        "<f4",
    )
    indices = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], "<u4")
    return dict(
        schema="flightrl.shared_forest.v1",
        units="m",
        up="z",
        quaternionOrder="xyzw",
        wind_m_s=[0, 0, 0],
        vertices=base64.b64encode(vertices).decode(),
        indices=base64.b64encode(indices).decode(),
        triangleCount=4,
        bodies=[
            dict(
                id=str(i),
                vehicle="fpv",
                position=[-2, i - 1, 1.5],
                quaternion=[0, 0, 0, 1],
                halfExtents=[0.0925, 0.106, 0.032],
                mass=0.377,
            )
            for i in range(3)
        ]
        + [
            dict(
                id="prop",
                position=[0, 0, 1],
                quaternion=[0, 0, 0, 1],
                halfExtents=[0.15] * 3,
                mass=1,
            )
        ],
    )


def test_mesh_units_and_indices_are_checked():
    payload = scene()
    payload["up"] = "y"
    with pytest.raises(ValueError, match="Z-up"):
        decode_scene(payload)
    payload = scene()
    payload["indices"] = base64.b64encode(np.array([0, 1, 999] * 4, "<u4")).decode()
    with pytest.raises(ValueError, match="indices"):
        decode_scene(payload)


def test_gravity_contacts_and_hover_share_one_world():
    world = ContactWorld(scene())
    for _ in range(150):
        world.step()
    assert world.positions[3, 2] == pytest.approx(0.15, abs=0.006)
    assert world.p[:, 2] == pytest.approx([1.5] * 3, abs=0.002)
    assert world.total_contacts > 0
    assert world.ticks == 150


def test_ccd_stops_fast_body_at_rendered_thin_wall():
    world = ContactWorld(scene())
    world.world.set_linear_velocity(world.handles[3], 200, 0, 0)
    world.step()
    assert world.positions[3, 0] < 2
    assert world.positions[3, 0] > 1.5


def test_surface_rays_use_metres_and_up_axis():
    world = ContactWorld(scene())
    result = world.rays([[1, 1, 3], [0, -2, 2]], [[0, 0, -1], [1, 0, 0]], 4)
    assert result["fraction"] == pytest.approx([0.75, 0.5], abs=1e-5)
    assert result["normal"][0] == pytest.approx([0, 0, 1], abs=1e-5)


def test_fixed_steps_repeat_and_contact_changes_motion():
    a, b = ContactWorld(scene()), ContactWorld(scene())
    for _ in range(70):
        a.step()
        b.step()
    np.testing.assert_allclose(a.positions, b.positions, atol=1e-7)
    assert np.linalg.norm(a.world.get_velocity(a.handles[3])) < 0.02


def test_particles_follow_moving_support_and_conserve_dust():
    from flightrl.realism.particles import Particles

    world = ContactWorld(scene())
    particles = Particles(world)
    body = world.handles[3]
    particles.p[0] = [0, 0, 1.152]
    particles.anchor(np.array([0]), np.array([body]))
    world.world.set_transform(body, [0.5, 0, 1], [0, 0, 0, 1])
    world.sync()
    particles.step()
    assert particles.p[0] == pytest.approx([0.5, 0, 1.152], abs=1e-5)
    particles.rain = True
    for _ in range(100):
        world.step()
        particles.step()
    counts = particles.record()
    assert counts["rain_impacts"] > 0
    assert counts["rain_emitted"] >= 320
    assert (
        counts["dust_airborne"] + counts["dust_settled"] + counts["dust_escaped"]
        == 1024
    )
    assert np.isfinite(particles.p).all()


def test_rays_reject_mismatched_arrays_before_native_access():
    world = ContactWorld(scene())
    with pytest.raises(ValueError, match="matching"):
        world.rays([[0, 0, 1]], [])


def test_demonstration_holds_position_and_stays_low_for_dust():
    from flightrl.realism.demo import DemoFlight
    from flightrl.realism.particles import Particles

    world = ContactWorld(scene())
    world.ambient_wind[:] = [0.12, 0.04, 0]
    world.wind[:] = world.ambient_wind
    demo = DemoFlight(world)
    particles = Particles(world)
    for _ in range(500):
        world.actions[:] = demo.controls()
        world.step()
        particles.step()
    np.testing.assert_allclose(world.p, demo.home, atol=0.04)
    demo.started = world.ticks * world.dt
    low = []
    for _ in range(1500):
        world.actions[:] = demo.controls(dust=True)
        world.step()
        particles.step()
        if world.ticks > 1500:
            low.append(world.p[:, 2].copy())
    assert np.min(low) > 0.35 and np.max(low) < 0.6
    assert particles.resuspended > 0
    assert np.max(np.abs(world.p[:, :2] - demo.home[:, :2])) < 0.6
