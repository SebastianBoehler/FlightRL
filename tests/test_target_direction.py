from __future__ import annotations

import numpy as np

from flightrl.hardware.avoidance_policy import RangerReading
from flightrl.hardware.target_direction import (
    TargetDirectionConfig,
    cruise_vector,
    keepout_pressure,
    target_direction_command,
)


def test_target_direction_cruises_when_clear() -> None:
    command = target_direction_command(
        RangerReading(front_m=3.0, back_m=3.0, left_m=3.0, right_m=3.0, up_m=2.0, zrange_m=0.5),
        TargetDirectionConfig(direction_deg=0.0, target_speed_m_s=0.2, max_speed_m_s=0.5),
    )

    assert np.isclose(command.vx_m_s, 0.2)
    assert abs(command.vy_m_s) < 1e-6
    assert command.zdistance_m == 0.5


def test_target_direction_moves_away_from_front_obstacle() -> None:
    command = target_direction_command(
        RangerReading(front_m=0.15, back_m=3.0, left_m=3.0, right_m=3.0, up_m=2.0, zrange_m=0.5),
        TargetDirectionConfig(direction_deg=0.0, target_speed_m_s=0.2, avoidance_speed_m_s=0.8, max_speed_m_s=0.8),
    )

    assert command.vx_m_s < 0.0


def test_target_direction_clips_combined_vector_norm() -> None:
    command = target_direction_command(
        RangerReading(front_m=3.0, back_m=0.15, left_m=3.0, right_m=0.15, up_m=2.0, zrange_m=0.5),
        TargetDirectionConfig(direction_deg=45.0, target_speed_m_s=0.4, avoidance_speed_m_s=0.8, max_speed_m_s=0.5),
    )

    assert np.linalg.norm([command.vx_m_s, command.vy_m_s]) <= 0.500001


def test_keepout_pressure_is_zero_outside_clearance() -> None:
    assert keepout_pressure(2.0, clearance_m=1.0, hard_clearance_m=0.1) == 0.0


def test_cruise_vector_uses_degrees() -> None:
    vx, vy = cruise_vector(90.0, 0.3)

    assert abs(vx) < 1e-6
    assert np.isclose(vy, 0.3)
