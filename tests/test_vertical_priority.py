from __future__ import annotations

from flightrl.hardware.avoidance_policy import AvoidanceCommand, RangerReading
from flightrl.hardware.vertical_priority import apply_vertical_priority


def test_vertical_priority_suppresses_soft_horizontal_escape_when_top_close() -> None:
    command = AvoidanceCommand(0.24, -0.16, 0.0, 0.50)
    reading = RangerReading(front_m=0.80, back_m=0.50, left_m=0.55, right_m=0.70, up_m=0.20, zrange_m=0.50)

    adjusted, active = apply_vertical_priority(
        command,
        reading,
        None,
        top_clearance_m=0.75,
        bottom_clearance_m=0.35,
        horizontal_escape_clearance_m=0.34,
        horizontal_hard_ttc_s=0.20,
    )

    assert active is True
    assert adjusted.vx_m_s == 0.0
    assert adjusted.vy_m_s == 0.0
    assert adjusted.zdistance_m == command.zdistance_m


def test_vertical_priority_preserves_escape_when_horizontal_obstacle_is_close() -> None:
    command = AvoidanceCommand(0.24, -0.16, 0.0, 0.50)
    reading = RangerReading(front_m=0.80, back_m=0.30, left_m=0.55, right_m=0.70, up_m=0.20, zrange_m=0.50)

    adjusted, active = apply_vertical_priority(
        command,
        reading,
        None,
        top_clearance_m=0.75,
        bottom_clearance_m=0.35,
        horizontal_escape_clearance_m=0.34,
        horizontal_hard_ttc_s=0.20,
    )

    assert active is False
    assert adjusted == command


def test_vertical_priority_preserves_escape_when_horizontal_ttc_is_hard() -> None:
    command = AvoidanceCommand(-0.24, 0.0, 0.0, 0.50)
    reading = RangerReading(front_m=0.60, back_m=1.50, left_m=1.50, right_m=1.50, up_m=0.20, zrange_m=0.50)
    range_rate = RangerReading(front_m=-4.0, back_m=0.0, left_m=0.0, right_m=0.0, up_m=0.0, zrange_m=0.0)

    adjusted, active = apply_vertical_priority(
        command,
        reading,
        range_rate,
        top_clearance_m=0.75,
        bottom_clearance_m=0.35,
        horizontal_escape_clearance_m=0.34,
        horizontal_hard_ttc_s=0.20,
    )

    assert active is False
    assert adjusted == command


def test_vertical_priority_also_handles_bottom_pressure() -> None:
    command = AvoidanceCommand(0.18, 0.12, 0.0, 0.50)
    reading = RangerReading(front_m=0.80, back_m=0.70, left_m=0.90, right_m=0.70, up_m=2.00, zrange_m=0.25)

    adjusted, active = apply_vertical_priority(
        command,
        reading,
        None,
        top_clearance_m=0.75,
        bottom_clearance_m=0.35,
        horizontal_escape_clearance_m=0.34,
        horizontal_hard_ttc_s=0.20,
    )

    assert active is True
    assert adjusted.vx_m_s == 0.0
    assert adjusted.vy_m_s == 0.0
