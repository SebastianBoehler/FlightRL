"""Physical and observation boundaries of the first mixed-robot adapter."""

import cv2
import mujoco as mj
import numpy as np
import pytest
from flightrl.robotics.world import RobotWorld, SOURCE
from flightrl.robotics.spec import import_robot
from flightrl.robotics.sensing import target_from_pixels, DICTIONARY, decode


def run_wheels(world, commands, seconds=2):
    world.wheels[:] = commands
    for _ in range(round(seconds / world.dt)):
        world.step()
    return world.data.xpos[world.rover].copy()


def test_contact_drives_rover_but_free_wheels_do_not():
    grounded = RobotWorld()
    p = run_wheels(grounded, [3, 3])
    assert p[0] > -1.5
    assert abs(p[1] - 1.2) < 0.03
    airborne = RobotWorld()
    airborne.model.opt.gravity[:] = 0
    airborne.model.geom_contype[:] = 0
    airborne.model.geom_conaffinity[:] = 0
    p = run_wheels(airborne, [3, 3])
    assert abs(p[0] + 2) < 0.03
    assert np.max(np.abs(grounded.data.actuator_force)) <= 3.001


def test_wall_contact_blocks_driven_rover():
    world = RobotWorld(
        [
            dict(
                name="barrier",
                position=[-1.3, 1.2, 0.4],
                half_extents=[0.1, 0.8, 0.4],
                color=[0.5, 0.5, 0.5, 1],
            )
        ]
    )
    p = run_wheels(world, [5, 5], 3)
    assert -1.8 < p[0] < -1.5
    assert world.data.ncon > 0


def test_robot_description_contains_actual_mechanics_and_mounts():
    world = RobotWorld()
    rover = world.specs[1]
    assert sum(link.mass_kg for link in rover.links) == pytest.approx(8.21)
    assert len(rover.actuators) == 2
    assert rover.actuators[0]["force_range"] == [-3, 3]
    assert len(rover.source_sha256) == 64
    with pytest.raises(ValueError, match="Missing robot"):
        import_robot(world.model, SOURCE, "absent")
    camera = np.array(world.cameras()["positions"][1]) + [0.035, 0, 0.012]
    site = mj.mj_name2id(world.model, mj.mjtObj.mjOBJ_SITE, "rover_camera")
    np.testing.assert_allclose(camera, world.data.site_xpos[site])


def test_fiducial_measurement_requires_visible_correct_id_and_metric_depth():
    image = np.full((384, 512, 3), 255, np.uint8)
    tag = cv2.aruco.generateImageMarker(DICTIONARY, 17, 100)
    image[142:242, 206:306] = tag[..., None]
    depth = np.full((384, 512), 2, np.float32)
    measurement = target_from_pixels(image, depth, 17)
    np.testing.assert_allclose(measurement, [2, 0, 0], atol=0.01)
    assert target_from_pixels(image, depth, 23) is None
    image[:] = 255
    assert target_from_pixels(image, depth, 17) is None
    with pytest.raises(ValueError, match="Incomplete"):
        decode(b"invalid")


def test_one_owner_drone_settles_on_same_ground():
    world = RobotWorld()
    world.actions[0] = -1
    for _ in range(1500):
        world.step()
    assert 0.02 < world.data.xpos[world.drone, 2] < 0.06
    assert np.isfinite(world.data.qpos).all()


def test_report_loss_holds_rover_until_delivery():
    from flightrl.robotics.industrial import equipment
    from flightrl.robotics.mission import InspectionMission

    items, targets = equipment()
    world = RobotWorld(items)
    mission = InspectionMission(world, targets)
    world.data.qpos[:3] = np.array(targets[0]["position"]) - [1.085, 0, 0.012]
    world.data.qpos[7:10] = np.array(targets[1]["position"]) - [0.77, 0, 0.14]
    mj.mj_forward(world.model, world.data)
    frames = []
    for ident, distance in ((17, 1.05), (23, 0.55)):
        rgb = np.full((384, 512, 3), 255, np.uint8)
        rgb[142:242, 206:306] = cv2.aruco.generateImageMarker(DICTIONARY, ident, 100)[
            ..., None
        ]
        frames.append([(rgb, np.full((384, 512), distance, np.float32))])
    mission.link = False
    from flightrl.robotics.environment import RobotEnvironment
    fixture = RobotEnvironment()
    fixture.world = world
    for i in range(10):
        mission.observe(frames, {**fixture.state(), "time_s": i * 0.1})
    assert mission.phase == ["hold", "wait"]
    assert mission.report_received is None
    assert all(mission.correct.values())
    mission.link = True
    mission.observe(frames, {**fixture.state(), "time_s": 1.1})
    assert mission.phase == ["hold", "dock"]
    assert mission.report_received == 1.1


def test_docking_marker_support_does_not_occlude_its_face():
    from flightrl.robotics.industrial import equipment

    world = RobotWorld(equipment()[0])
    hit = np.array([-1], np.int32)
    distance = mj.mj_ray(
        world.model,
        world.data,
        np.array([-2.0, 1.2, 0.31]),
        np.array([-1.0, 0, 0]),
        None,
        1,
        world.rover,
        hit,
    )
    assert distance == pytest.approx(1.285, abs=0.001)
    assert mj.mj_id2name(world.model, mj.mjtObj.mjOBJ_GEOM, int(hit[0])) == "marker_42"


def test_camera_mount_frame_survives_attitude_change():
    world = RobotWorld()
    mj.mju_euler2Quat(world.data.qpos[3:7], np.array([0.3, -0.2, 1.1]), "xyz")
    mj.mj_forward(world.model, world.data)
    position = np.array(world.cameras()["positions"][0])
    rotation = world.data.xmat[world.drone].reshape(3, 3)
    site = mj.mj_name2id(world.model, mj.mjtObj.mjOBJ_SITE, "drone_camera")
    np.testing.assert_allclose(
        position + rotation @ [0.035, 0, 0.012], world.data.site_xpos[site], atol=1e-10
    )


def test_wheel_contact_step_refinement():
    coarse, fine = RobotWorld(), RobotWorld()
    fine.dt = 0.001
    fine.model.opt.timestep = fine.dt
    np.testing.assert_allclose(
        run_wheels(coarse, [3, 3]), run_wheels(fine, [3, 3]), atol=0.015
    )
