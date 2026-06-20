from __future__ import annotations

from flightrl.hardware.avoidance_close_escape import apply_close_escape_correction, body_velocity_m_s
from flightrl.hardware.avoidance_policy import AvoidanceCommand, RangerReading


def test_right_side_override_forces_left_escape() -> None:
    command = AvoidanceCommand(0.0, -0.05, 0.0, 0.5)
    reading = RangerReading(front_m=1.0, back_m=1.0, left_m=1.0, right_m=0.18, up_m=1.0, zrange_m=0.5)

    corrected, status = apply_close_escape_correction(
        command,
        reading,
        {"stateEstimate.vx": 0.0, "stateEstimate.vy": -0.2, "stabilizer.yaw": 0.0},
        clearance_m=0.32,
        min_speed_m_s=0.35,
        brake_gain=0.5,
        brake_max_m_s=0.2,
    )

    assert corrected.vy_m_s > 0.35
    assert status.closest_side == "right"
    assert status.override_active is True
    assert status.brake_active is True


def test_front_close_brakes_forward_body_velocity() -> None:
    command = AvoidanceCommand(-0.2, 0.0, 0.0, 0.5)
    reading = RangerReading(front_m=0.2, back_m=1.0, left_m=1.0, right_m=1.0, up_m=1.0, zrange_m=0.5)

    corrected, status = apply_close_escape_correction(
        command,
        reading,
        {"stateEstimate.vx": 0.4, "stateEstimate.vy": 0.0, "stabilizer.yaw": 0.0},
        clearance_m=0.32,
        min_speed_m_s=0.25,
        brake_gain=0.5,
        brake_max_m_s=0.3,
    )

    assert corrected.vx_m_s < -0.25
    assert status.brake_active is True


def test_body_velocity_rotates_world_velocity_by_yaw() -> None:
    body_vx, body_vy = body_velocity_m_s(
        {"stateEstimate.vx": 1.0, "stateEstimate.vy": 0.0, "stabilizer.yaw": 90.0}
    )

    assert abs(body_vx) < 1e-6
    assert body_vy < -0.99
