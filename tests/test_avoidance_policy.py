from __future__ import annotations

import numpy as np
import torch

from flightrl.hardware.avoidance_policy import (
    AvoidanceCommand,
    RangerAvoidancePolicy,
    RangerReading,
    clip_horizontal_norm,
    command_row,
    command_from_model,
    min_horizontal_range_m,
    min_horizontal_ttc_s,
    normalize_reading,
    reactive_clearance_command,
    smooth_command,
    teacher_command,
    vertical_velocity_from_clearance,
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


def test_command_from_model_uses_configurable_speed_clip() -> None:
    model = RangerAvoidancePolicy(hidden_size=8)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.net[-1].bias.data = torch.tensor([2.0, -2.0, 90.0, 0.5])
    reading = RangerReading(front_m=1.0, back_m=1.0, left_m=1.0, right_m=1.0, up_m=2.0, zrange_m=0.5)

    command = command_from_model(model, reading, max_speed_m_s=1.1, max_yawrate_deg_s=30.0)

    assert np.isclose(command.vx_m_s, 1.1)
    assert np.isclose(command.vy_m_s, -1.1)
    assert np.isclose(command.yawrate_deg_s, 30.0)
    assert np.isclose(command.zdistance_m, 0.5)


def test_clip_horizontal_norm_limits_vector_speed() -> None:
    command = clip_horizontal_norm(AvoidanceCommand(0.8, 0.8, 90.0, 0.5), max_speed=0.8, max_yawrate=45.0)

    assert np.linalg.norm([command.vx_m_s, command.vy_m_s]) <= 0.800001
    assert command.yawrate_deg_s == 45.0


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


def test_vertical_clearance_velocity_reacts_without_height_target() -> None:
    open_reading = RangerReading(front_m=2.0, back_m=2.0, left_m=2.0, right_m=2.0, up_m=2.0, zrange_m=0.5)
    top_close = RangerReading(front_m=2.0, back_m=2.0, left_m=2.0, right_m=2.0, up_m=0.2, zrange_m=0.5)
    bottom_close = RangerReading(front_m=2.0, back_m=2.0, left_m=2.0, right_m=2.0, up_m=2.0, zrange_m=0.2)

    assert vertical_velocity_from_clearance(open_reading) == 0.0
    assert vertical_velocity_from_clearance(top_close) < 0.0
    assert vertical_velocity_from_clearance(bottom_close) > 0.0


def test_vertical_clearance_floor_guard_wins_over_sustained_top_pressure() -> None:
    top_only = RangerReading(front_m=2.0, back_m=2.0, left_m=2.0, right_m=2.0, up_m=0.18, zrange_m=0.5)
    squeezed = RangerReading(front_m=2.0, back_m=2.0, left_m=2.0, right_m=2.0, up_m=0.18, zrange_m=0.25)

    assert vertical_velocity_from_clearance(top_only) < 0.0
    assert vertical_velocity_from_clearance(squeezed) > 0.0


def test_reactive_single_hard_clearance_dominates_opposite_sensor() -> None:
    command = reactive_clearance_command(RangerReading(front_m=0.08, back_m=2.0, left_m=2.0, right_m=2.0, up_m=2.0, zrange_m=0.45))

    assert command.vx_m_s == -0.25
    assert abs(command.vy_m_s) < 1e-6


def test_reactive_front_back_pinch_escapes_toward_open_side() -> None:
    command = reactive_clearance_command(RangerReading(front_m=0.08, back_m=0.20, left_m=2.0, right_m=0.6, up_m=2.0, zrange_m=0.45))

    assert command.vx_m_s < 0.0
    assert command.vy_m_s > 0.15
    assert np.linalg.norm([command.vx_m_s, command.vy_m_s]) <= 0.250001


def test_reactive_left_right_pinch_escapes_toward_open_side() -> None:
    command = reactive_clearance_command(RangerReading(front_m=2.0, back_m=0.6, left_m=0.08, right_m=0.20, up_m=2.0, zrange_m=0.45))

    assert command.vx_m_s > 0.15
    assert command.vy_m_s < 0.0
    assert np.linalg.norm([command.vx_m_s, command.vy_m_s]) <= 0.250001


def test_reactive_left_right_tunnel_prioritizes_longitudinal_escape() -> None:
    command = reactive_clearance_command(RangerReading(front_m=2.0, back_m=0.6, left_m=0.28, right_m=0.25, up_m=2.0, zrange_m=0.45))

    assert command.vx_m_s > 0.15
    assert abs(command.vy_m_s) < 0.04


def test_reactive_left_right_tunnel_keeps_hard_side_escape() -> None:
    command = reactive_clearance_command(RangerReading(front_m=2.0, back_m=0.6, left_m=0.08, right_m=0.25, up_m=2.0, zrange_m=0.45))

    assert command.vx_m_s > 0.10
    assert command.vy_m_s < -0.08


def test_reactive_corner_escape_is_diagonal() -> None:
    command = reactive_clearance_command(RangerReading(front_m=0.12, back_m=2.0, left_m=2.0, right_m=0.12, up_m=2.0, zrange_m=0.45))

    assert command.vx_m_s < -0.1
    assert command.vy_m_s > 0.1


def test_reactive_ttc_anticipates_fast_front_closure() -> None:
    reading = RangerReading(front_m=0.8, back_m=2.0, left_m=2.0, right_m=2.0, up_m=2.0, zrange_m=0.45)
    rate = RangerReading(front_m=-2.0, back_m=0.0, left_m=0.0, right_m=0.0, up_m=0.0, zrange_m=0.0)

    command = reactive_clearance_command(reading, range_rate_m_s=rate, ttc_horizon_s=0.7, ttc_hard_s=0.15, max_speed_m_s=0.5)

    assert command.vx_m_s < -0.1
    assert min_horizontal_ttc_s(reading, rate) == 0.4


def test_min_horizontal_range_ignores_vertical_sensors() -> None:
    reading = RangerReading(front_m=0.4, back_m=1.0, left_m=0.3, right_m=0.8, up_m=0.05, zrange_m=0.02)

    assert min_horizontal_range_m(reading) == 0.3


def test_smooth_command_limits_step_size() -> None:
    previous = AvoidanceCommand(vx_m_s=0.0, vy_m_s=0.0, yawrate_deg_s=0.0, zdistance_m=0.5)
    target = AvoidanceCommand(vx_m_s=-0.25, vy_m_s=0.2, yawrate_deg_s=30.0, zdistance_m=0.7)

    command = smooth_command(
        target,
        previous,
        alpha=1.0,
        max_speed_step_m_s=0.03,
        max_yawrate_step_deg_s=6.0,
        max_zdistance_step_m=0.04,
    )

    assert command.vx_m_s == -0.03
    assert command.vy_m_s == 0.03
    assert command.yawrate_deg_s == 6.0
    assert command.zdistance_m == 0.54
