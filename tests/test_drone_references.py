"""Dimensioned assets must not silently change mass, timing or policy compatibility."""

import copy
import mujoco as mj
import numpy as np
import pytest
from test_realism_physics import scene
from flightrl.fleet.vehicles import VEHICLES
from flightrl.robotics.drone_asset import drone_model
from flightrl.robotics.world import RobotWorld, SOURCE
from flightrl.realism.physics import ContactWorld
from flightrl.realism.demo import DemoFlight
from flightrl.realism.particles import Particles
from flightrl.realism.session import Session


def agriculture_scene():
    payload = scene()
    model = drone_model("agriculture")
    for i, body in enumerate(payload["bodies"][:3]):
        body.update(vehicle="agriculture", mass=model["mass_kg"],
                    halfExtents=(np.array(model["dimensions_m"]) / 2).tolist(),
                    position=[-3, (i - 1) * 4, 2.3],
                    model={k: v for k, v in model.items() if k != "parts"})
    return payload


def test_fpv_visual_asset_preserves_native_mechanics_and_sensor_mount():
    world = RobotWorld()
    original = mj.MjModel.from_xml_path(str(SOURCE))
    drone = world.drone
    np.testing.assert_allclose(world.model.body_mass[drone], original.body_mass[drone])
    np.testing.assert_allclose(world.model.body_inertia[drone], original.body_inertia[drone])
    assert world.model.body_mass[drone] == pytest.approx(.377)
    description = world.render_description()
    visuals = [g for g in description["geometries"] if g["name"].startswith("drone_visual_")]
    assert len(visuals) == 10 and all(g["type"] == "mesh" for g in visuals)
    assert not any(g["name"] == "drone_chassis" for g in description["geometries"])
    assert world.specs[0].source_sha256 != world.specs[1].source_sha256
    np.testing.assert_allclose(world.model.site_pos[0], [.035, 0, .012])


@pytest.mark.parametrize("kind", ["fpv", "agriculture"])
def test_authored_visuals_fit_the_physical_envelope(kind):
    model = drone_model(kind)
    half = np.array(model["dimensions_m"]) / 2
    for part in model["parts"]:
        vertices = np.array(part["vertices"]).reshape(-1, 3) + part["position"]
        assert np.isfinite(vertices).all()
        assert (np.abs(vertices).max(axis=0) <= half + .001).all(), part["name"]
        if part["name"].startswith("rotor_"):
            sweep = np.linalg.norm(np.array(part["vertices"]).reshape(-1, 3)[:, :2], axis=1).max()
            assert (np.abs(part["position"][:2]) + sweep <= half[:2] + .001).all()


def test_agricultural_mass_response_and_safe_dust_clearance():
    world = ContactWorld(agriculture_scene())
    try:
        np.testing.assert_allclose(world.params, np.tile(VEHICLES["agriculture"].physics(), (3, 1)))
        assert world.params[0, 0] == 32
        demo = DemoFlight(world)
        for _ in range(600):
            world.actions[:] = demo.controls(dust=True)
            world.step()
        np.testing.assert_allclose(world.p[:, 2], [.99] * 3, atol=.08)
        assert np.isfinite(world.p).all()
        assert world.total_contacts < 10  # Falling prop only; separated aircraft hover freely.
    finally:
        world.world.close()


def test_agricultural_dust_starts_below_aircraft_not_on_rotor_proxy():
    world = ContactWorld(agriculture_scene())
    try:
        particles = Particles(world)
        assert (particles.p[particles.kind == 0, 2] < .01).all()
        assert not np.isin(particles.support[particles.kind == 0], world.handles[:3]).any()
    finally:
        world.world.close()


def test_mismatched_visual_and_physical_references_are_rejected():
    payload = agriculture_scene()
    payload["bodies"][0]["mass"] = .377
    with pytest.raises(ValueError, match="mass"):
        ContactWorld(payload)
    payload = agriculture_scene()
    payload["bodies"][0]["model"] = copy.deepcopy(payload["bodies"][0]["model"])
    payload["bodies"][0]["model"]["camera_offset_m"][0] += .1
    with pytest.raises(ValueError, match="Displayed"):
        ContactWorld(payload)


def test_fpv_policy_is_not_applied_to_agricultural_aircraft():
    session = Session.__new__(Session)
    session.world = ContactWorld(agriculture_scene())
    session.mode = "hover"
    try:
        with pytest.raises(ValueError, match="FPV policy"):
            session.set_mode("policy")
        assert session.mode == "hover"
    finally:
        session.world.world.close()
