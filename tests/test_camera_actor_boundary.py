import numpy as np
import pytest
from flightrl.fleet.camera_policy.sensors import CameraPacket
from flightrl.fleet.camera_policy.contract import contract
from flightrl.fleet.camera_policy.teacher import decisions


def packet():
    rgb = np.zeros((3, 48, 64, 3), np.uint8)
    for i, color in enumerate([(220, 30, 30), (30, 30, 220), (30, 220, 30)]):
        rgb[i, 20:28, 28:36] = color
    return CameraPacket(
        rgb,
        np.ones((3, 48, 64), np.float32),
        np.zeros((3, 9), np.float32),
        np.eye(3, dtype=np.float32),
        np.zeros((3, 6), np.float32),
        0,
        0.0,
    )


def test_actor_packet_cannot_receive_map_pose_or_goals():
    p = packet()
    p.validate()
    with pytest.raises(TypeError):
        CameraPacket(**{**p.__dict__, "positions": np.zeros((3, 3))})
    fields = [s["name"] for s in contract()["observation"]["signals"]]
    assert not set(fields) & {"position", "target", "waypoints", "geometry"}
    assert contract()["action"]["fields"] == [
        "collective_thrust",
        "roll_rate",
        "pitch_rate",
        "yaw_rate",
    ]


def test_visual_reports_require_pixels_and_confirmation_message():
    p = packet()
    assert decisions(p)[2].tolist() == [1, 1, 0]
    p.messages[:, :4] = 1
    assert decisions(p)[2].tolist() == [1, 1, 1]
    p.rgb[:] = 0
    assert decisions(p)[2].tolist() == [0, 0, 0]


def test_nonfinite_depth_is_not_a_valid_observation():
    p = packet()
    p.depth[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        p.validate()
