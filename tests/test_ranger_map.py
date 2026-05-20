from __future__ import annotations

from flightrl.hardware.ranger_map import trajectory_from_rows


def test_trajectory_from_rows_keeps_pose_samples() -> None:
    rows = [
        {
            "host_time_s": "1.5",
            "stateEstimate.x": "0.1",
            "stateEstimate.y": "-0.2",
            "stateEstimate.z": "0.3",
            "stabilizer.roll": "1.0",
            "stabilizer.pitch": "2.0",
            "stabilizer.yaw": "3.0",
        }
    ]
    poses = trajectory_from_rows(rows)

    assert len(poses) == 1
    assert poses[0].time_s == 1.5
    assert poses[0].x_m == 0.1
    assert poses[0].yaw_deg == 3.0
