from __future__ import annotations

import numpy as np

from flightrl.hardware.avoidance_policy import (
    AvoidanceCommand,
    RangerAvoidancePolicy,
    RangerReading,
    command_row,
    normalize_reading,
    reactive_clearance_command,
    teacher_command,
    vertical_velocity_from_height_error,
)


def test_teacher_moves_back_when_front_is_close() -> None:
    command = teacher_command(RangerReading(front_m=0.2, back_m=1.5, left_m=1.5, right_m=1.5, up_m=2.0, zrange_m=0.45))

    assert command.vx_m_s < 0.0
    assert abs(command.vy_m_s) < 1e-6
    assert command.zdistance_m == 0.45


def test_teacher_moves_away_from_right_wall() -> None:
    command = teacher_command(RangerReading(front_m=1.5, back_m=1.5, left_m=1.5, right_m=0.25, up_m=2.0, zrange_m=0.45))

    assert command.vy_m_s > 0.0


def test_policy_forward_shape_matches_hover_command() -> None:
    model = RangerAvoidancePolicy(hidden_size=16)
    observation = normalize_reading(
        RangerReading(front_m=1.0, back_m=1.0, left_m=1.0, right_m=1.0, up_m=2.0, zrange_m=0.45)
    )

    output = model(np.asarray([observation], dtype=np.float32))

    assert output.shape == (1, 4)


def test_command_row_serializes_slots_dataclass() -> None:
    row = command_row(AvoidanceCommand(vx_m_s=0.1, vy_m_s=-0.2, yawrate_deg_s=3.0, zdistance_m=0.45))

    assert row == {
        "vx_m_s": 0.1,
        "vy_m_s": -0.2,
        "yawrate_deg_s": 3.0,
        "zdistance_m": 0.45,
    }


def test_reactive_clearance_moves_away_from_each_side() -> None:
    front = reactive_clearance_command(RangerReading(front_m=0.1, back_m=2.0, left_m=2.0, right_m=2.0, up_m=2.0, zrange_m=0.45))
    right = reactive_clearance_command(RangerReading(front_m=2.0, back_m=2.0, left_m=2.0, right_m=0.1, up_m=2.0, zrange_m=0.45))
    left = reactive_clearance_command(RangerReading(front_m=2.0, back_m=2.0, left_m=0.1, right_m=2.0, up_m=2.0, zrange_m=0.45))

    assert front.vx_m_s < -0.2
    assert right.vy_m_s > 0.2
    assert left.vy_m_s < -0.2


def test_vertical_velocity_uses_height_error() -> None:
    command = reactive_clearance_command(
        RangerReading(front_m=2.0, back_m=2.0, left_m=2.0, right_m=2.0, up_m=2.0, zrange_m=0.12),
        target_height_m=0.45,
    )
    vz = vertical_velocity_from_height_error(command, RangerReading(2.0, 2.0, 2.0, 2.0, 2.0, 0.12))

    assert command.zdistance_m > 0.45
    assert vz > 0.0


def test_reactive_hard_clearance_dominates_opposite_sensor() -> None:
    command = reactive_clearance_command(RangerReading(front_m=0.08, back_m=0.20, left_m=2.0, right_m=2.0, up_m=2.0, zrange_m=0.45))

    assert command.vx_m_s == -0.25
