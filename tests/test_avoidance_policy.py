from __future__ import annotations

import numpy as np

from flightrl.hardware.avoidance_policy import (
    AvoidanceCommand,
    RangerAvoidancePolicy,
    RangerReading,
    command_row,
    normalize_reading,
    teacher_command,
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
