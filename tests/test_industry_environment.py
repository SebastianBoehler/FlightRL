"""Sensor, geometry and policy boundaries for varied industrial work orders."""

import cv2
import numpy as np
from flightrl.robotics.environment import RobotEnvironment
from flightrl.robotics.sensing import DICTIONARY
from flightrl.robotics.visual_control import (
    inspect_marker,
    features,
    teacher,
    supervise,
)
from flightrl.robotics.utility_site import utility_site


def optical_fixture():
    rgb = np.full((384, 512, 3), 235, np.uint8)
    rgb[142:242, 206:306] = cv2.aruco.generateImageMarker(DICTIONARY, 17, 100)[
        ..., None
    ]
    rgb[94:105, 249:263] = [0, 220, 0]
    y, x = np.mgrid[:384, :512]
    f = 384 / (2 * np.tan(np.deg2rad(63) / 2))
    rays = np.stack([np.ones_like(x), -(x - 255.5) / f, -(y - 191.5) / f], -1)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    normal = np.array([-np.cos(0.2), np.sin(0.2), 0])
    distance = (-2 * np.cos(0.2) / (rays @ normal)).astype(np.float32)
    return rgb, distance, normal


def test_marker_plane_and_indicator_are_measured_from_pixels():
    rgb, depth, normal = optical_fixture()
    m = inspect_marker(rgb, depth, 17)
    assert m["signal"] == 0
    np.testing.assert_allclose(m["normal"], normal, atol=0.003)
    assert inspect_marker(rgb, depth, 18) is None
    rgb[94:105, 249:263] = [220, 0, 0]
    assert inspect_marker(rgb, depth, 17)["signal"] == 1
    rgb[:] = 235
    assert inspect_marker(rgb, depth, 17) is None


def test_clearance_and_missing_depth_stop_motion():
    rgb, depth, _ = optical_fixture()
    m = inspect_marker(rgb, depth, 17)
    x = features(m, depth, np.zeros(9, np.float32), "drone")
    assert x.shape == (21,) and np.isfinite(x).all()
    assert teacher(m, depth, "drone")[0][1] > 0.1
    blocked = np.full_like(depth, 0.25)
    assert supervise([0.4, 0, 0, 0], blocked, "rover")[0] == 0
    missing = np.full_like(depth, np.nan)
    np.testing.assert_array_equal(
        supervise([0.4, 0.1, 0.1, 0.1], missing, "drone", False), 0
    )
    assert np.isfinite(features(None, missing, np.zeros(9), "rover")).all()


def test_real_layout_variation_and_exact_reset():
    a = RobotEnvironment(120, industry=True)
    desc = a.description()
    assert any("transformer" in g["name"] for g in desc["geometries"])
    assert any("production_" in x["name"] for x in utility_site(121)[0])
    original = a.state()["bodies"]["positions"]
    a.step()
    a.reset(120)
    np.testing.assert_allclose(a.state()["bodies"]["positions"], original)
    assert a.description() == desc
    assert utility_site(121)[1] != utility_site(120)[1]


def test_capture_delay_and_outage_have_explicit_validity():
    env = RobotEnvironment(0, industry=True)
    rig = env.sensor_rig
    frames = [
        [(np.zeros((24, 32, 3), np.uint8), np.full((24, 32), 2, np.float32))]
        for _ in range(2)
    ]
    assert rig.deliver(frames, dict(time_s=2.9, sequence=1)) is None
    delayed, state = rig.deliver(frames, dict(time_s=3.3, sequence=2))
    assert state["sequence"] == 1 and rig.valid == [False, False]
    assert np.isnan(delayed[0][0][1]).all()
    delayed, state = rig.deliver(frames, dict(time_s=3.4, sequence=3))
    assert state["sequence"] == 2 and rig.valid == [True, True]
    assert np.isfinite(delayed[0][0][1]).mean() > 0.9


def test_controller_observations_do_not_change_when_hidden_pose_changes():
    env = RobotEnvironment(0, industry=True)
    rgb, depth, _ = optical_fixture()
    target = env.targets[0]["id"]
    rgb[142:242, 206:306] = cv2.aruco.generateImageMarker(DICTIONARY, target, 100)[
        ..., None
    ]
    frames = [[(rgb, depth)], [(rgb, depth)]]
    state = env.state()
    env.mission.observe(frames, state)
    before = env.mission.commands.copy()
    env.world.data.qpos[:3] += [10, 5, 2]
    env.mission.observe(frames, state)
    np.testing.assert_allclose(env.mission.commands, before)


def test_visual_station_keeping_continues_after_inspection():
    env = RobotEnvironment(0, industry=True)
    rgb, depth, _ = optical_fixture()
    rgb[142:242, 206:306] = cv2.aruco.generateImageMarker(
        DICTIONARY, env.targets[0]["id"], 100
    )[..., None]
    env.mission.phase[0] = "hold"
    for _ in range(5):
        env.mission.observe([[(rgb, depth)], [(rgb, depth)]], env.state())
    assert env.mission.commands[0, 0] > 0.1
    assert env.mission.events == []


def test_equipment_collision_invalidates_reports_and_terminates(tmp_path):
    from flightrl.robotics.session import RobotSession

    session = RobotSession(tmp_path / "collision", 0, industry=True)
    session.mission.correct = {t["id"]: True for t in session.targets}
    assert session.mission.status()["success"]
    session.world.robot_collision_steps = 1
    session.step()
    assert not session.mission.status()["success"]
    assert session.paused and "contacted" in session.stop_reason
    rgb, depth, _ = optical_fixture()
    outcome = session.environment.observe([[(rgb, depth)]] * 2, session.state())
    assert outcome["terminated"]


def test_estimated_attitude_preserves_world_up():
    env = RobotEnvironment(0, industry=True)
    for i, body in enumerate((env.world.drone, env.world.rover)):
        assert abs(np.dot(env.sensor_rig.q[i], env.world.data.xquat[body])) > 0.999


def test_industry_rover_waits_for_camera_handover_then_uses_its_work_order():
    env = RobotEnvironment(0, industry=True)
    frames = []
    for target, distance in zip(env.targets[:2], (1.05, 2.0)):
        rgb, depth, _ = optical_fixture()
        rgb[142:242, 206:306] = cv2.aruco.generateImageMarker(
            DICTIONARY, target["id"], 100
        )[..., None]
        frames.append([(rgb, np.full_like(depth, distance))])
    env.mission.link = False
    for i in range(8):
        state = env.state()
        state["time_s"] = i * 0.1
        env.mission.observe(frames, state)
    handover = env.mission.handover.copy()
    assert handover["asset_id"] == env.targets[0]["id"]
    assert handover["observed_signal"] == 0
    assert len(handover["image_sha256"]) == 64
    assert env.mission.phase[1] == "await_task"
    np.testing.assert_array_equal(env.mission.commands[1], 0)
    assert env.mission.report_received is None
    env.mission.link = True
    state["time_s"] = 1.0
    env.mission.observe(frames, state)
    assert env.mission.active_targets[1] == handover["followup_marker"]
    assert env.mission.commands[1, 0] > 0.1
    assert env.mission.report_received == 1.0
