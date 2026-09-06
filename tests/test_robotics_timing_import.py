"""Causal recording, captured-pose scoring and reference-model preservation."""

import base64
import json
import time
from io import BytesIO

import cv2
import mujoco as mj
import numpy as np
import pytest
from mcap.reader import make_reader

from flightrl.robotics.environment import RobotEnvironment
from flightrl.robotics.model_asset import ARM_SOURCE
from flightrl.robotics.recording import RunRecorder, replay_capture
from flightrl.robotics.sensing import SIZES
from flightrl.robotics.session import RobotSession
from flightrl.robotics.world import RobotWorld


def packed_frames(value):
    pieces = []
    for _ in range(2):
        for w, h in SIZES:
            rgb = np.full((h, w, 4), value, np.uint8)
            rgb[..., 3] = 255
            pieces.extend([rgb.tobytes(), np.full((h, w), 2, "<f4").tobytes()])
    return b"".join(pieces)


def test_inspection_scores_the_pose_at_image_acquisition():
    env = RobotEnvironment(0, industry=True)
    target = env.targets[0]
    captured = env.state()
    expected = np.array(target["position"]) + 1.05 * np.array(target["approach"])
    captured["camera_poses"][0]["position_m"] = expected.tolist()
    env.world.data.qpos[:3] += 100
    mj.mj_forward(env.world.model, env.world.data)
    env.world.ticks = 200
    env.mission.record(
        0,
        "drone",
        target["id"],
        {"signal": target["signal"], "relative": np.array([1.05, 0, 0])},
        captured,
        np.zeros((2, 2, 3), np.uint8),
    )
    event = env.mission.events[0]
    assert event["verified"] and event["position_error_m"] < 1e-12
    assert event["time_s"] == 0 and event["decision_time_s"] == 0.4


def test_delay_uses_acquisition_age_and_rejects_clock_reversal():
    env = RobotEnvironment(0, industry=True)
    frames = [[(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32))]] * 2
    rig = env.sensor_rig
    assert rig.deliver(frames, dict(time_s=0, sequence=0), 0.02) is None
    assert rig.deliver(frames, dict(time_s=0.05, sequence=25), 0.07) is None
    _, state = rig.deliver(frames, dict(time_s=0.08, sequence=40), 0.16)
    assert state["sequence"] == 25
    assert state["observation_age_s"] == pytest.approx(0.11)
    assert state["skipped_captures"] == 1
    with pytest.raises(ValueError, match="increase"):
        rig.deliver(frames, dict(time_s=0.07, sequence=35), 0.2)


def test_saved_training_sample_and_mcap_preserve_causal_observation(tmp_path):
    session = RobotSession(tmp_path / "episode", 0, industry=True)
    try:
        for i, value in enumerate((37, 83)):
            session.world.ticks = i * 50
            state = session.state()
            session.pending = (i, state, time.perf_counter())
            session.meta = {"id": i}
            session.receive(packed_frames(value))
        sample = session.samples[0]
        assert sample["sequence"] == 0 and np.all(sample["rgb"] == 37)
        assert sample["decision_time_ns"] == 100_000_000
        feedback = session.state()["proprio"]
        session.step()
        session.paused = True
        session.save()
        with session.recorder.path.open("rb") as f:
            messages = list(make_reader(f).iter_messages())
        decisions = [
            json.loads(m.data) for _, c, m in messages if c.topic == "/decision"
        ]
        assert decisions[0]["capture_sequence"] == 0
        assert decisions[0]["decision_tick"] == 50
        actions = [
            json.loads(m.data) for _, c, m in messages if c.topic == "/actuation"
        ]
        assert actions[0]["source_capture_sequence"] == 0
        assert actions[0]["application_tick"] == 50
        assert actions[0]["feedback_tick"] == 50
        assert actions[0]["feedback_proprio"] == feedback
        image = next(m for _, c, m in messages if c.topic == "/drone/observed/rgbd/1")
        with np.load(BytesIO(image.data)) as stored:
            np.testing.assert_equal(stored["depth"], sample["depth"])
        replay = replay_capture(session.recorder.path, 50, 100_000_000)
        assert replay["state"]["time_s"] == 0.1 and len(replay["images"]) == 2
        pixels = cv2.imdecode(
            np.frombuffer(base64.b64decode(replay["images"]["drone"]), np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert np.all(pixels == 83)
        assert all(m.log_time == m.publish_time for _, _, m in messages)
        with pytest.raises(ValueError, match="not in this episode"):
            replay_capture(session.recorder.path, 999, 100_000_000)
    finally:
        session.recorder.finish()


def test_full_arm_preserves_reference_actuation_and_constraints():
    source = mj.MjModel.from_xml_path(str(ARM_SOURCE))
    world = RobotWorld(arm=True)
    m, arm = world.model, world.arm
    assert len(arm.joints) == source.njnt == 13
    assert len(arm.actuators) == source.nu == 8
    assert m.ntendon == source.ntendon and m.neq == source.neq
    np.testing.assert_allclose(m.actuator_gear[arm.actuators], source.actuator_gear)
    np.testing.assert_allclose(
        m.actuator_gainprm[arm.actuators], source.actuator_gainprm
    )
    np.testing.assert_allclose(
        m.actuator_biasprm[arm.actuators], source.actuator_biasprm
    )
    assert len(world.specs[-1].actuators) == 8
    assert world.specs[-1].actuators[-1]["transmission"] == mj.mjtTrn.mjTRN_TENDON
    start = arm.state()["position_rad"][0]
    values = arm.target.copy()
    values[0] = 0.25
    values[-1] = 180
    arm.command(values)
    for _ in range(1500):
        world.step()
    assert arm.state()["position_rad"][0] > start + 0.2
    assert np.isfinite(world.data.qpos).all() and not world.data.warning.number.any()
    for i in arm.actuators:
        assert (
            m.actuator_forcerange[i, 0] - 0.001
            <= world.data.actuator_force[i]
            <= m.actuator_forcerange[i, 1] + 0.001
        )
    with pytest.raises(ValueError, match="limits"):
        arm.command(values * 100)
    with pytest.raises(ValueError, match="finite"):
        arm.command([np.nan] * 8)


def test_rotated_wrist_camera_and_compiled_meshes_are_exported():
    world = RobotWorld(arm=True)
    desc = world.render_description()
    visual_meshes = {
        int(world.model.geom_dataid[i])
        for i in range(world.model.ngeom)
        if world.model.geom_type[i] == mj.mjtGeom.mjGEOM_MESH
        and world.model.geom_group[i] != 3
    }
    assert len(desc["cameras"]) == 3 and set(desc["meshes"]) == visual_meshes
    for geom in desc["geometries"]:
        if geom["type"] == "mesh":
            mesh = desc["meshes"][geom["mesh"]]
            assert max(mesh["indices"]) < len(mesh["vertices"]) // 3
    pose = world.cameras()
    q = np.array(pose["quaternions"][2])[[3, 0, 1, 2]]
    r = np.zeros(9)
    mj.mju_quat2Mat(r, q)
    site = world.camera_sites[2]
    np.testing.assert_allclose(r, world.data.site_xmat[site], atol=1e-10)
    np.testing.assert_allclose(
        np.array(pose["positions"][2]) + r.reshape(3, 3) @ [0.035, 0, 0.012],
        world.data.site_xpos[site],
        atol=1e-10,
    )


def test_recording_failure_is_explicit(tmp_path):
    recorder = RunRecorder(tmp_path, {}, {})
    recorder.submit("/bad", {"value": float("nan")}, 0, 0, 0)
    with pytest.raises(RuntimeError, match="Recording failed"):
        recorder.finish()
