from __future__ import annotations

import numpy as np

from flightrl.hardware.avoidance_policy import RangerReading
from flightrl.hardware.hold_policy import (
    HOLD_OBSERVATION_DIM,
    HoldState,
    RangerHoldPolicy,
    hold_command_array,
    normalize_hold_state,
    sample_hold_states,
    teacher_hold_command,
)


def test_hold_observation_shape_is_stable() -> None:
    state = sample_hold_states(1, np.random.default_rng(7))[0]

    observation = normalize_hold_state(state)

    assert observation.shape == (HOLD_OBSERVATION_DIM,)
    assert np.isfinite(observation).all()


def test_hold_teacher_moves_back_when_front_is_close() -> None:
    state = HoldState(
        ranges=RangerReading(front_m=0.08, back_m=2.0, left_m=2.0, right_m=2.0, up_m=2.0, zrange_m=0.45),
        x_m=0.0,
        y_m=0.0,
        z_m=0.45,
        vx_m_s=0.0,
        vy_m_s=0.0,
        vz_m_s=0.0,
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        gyro_x_rad_s=0.0,
        gyro_y_rad_s=0.0,
        gyro_z_rad_s=0.0,
        target_x_m=0.0,
        target_y_m=0.0,
        target_z_m=0.45,
    )

    command = teacher_hold_command(state)

    assert command.vx_m_s < -0.3
    assert abs(command.vy_m_s) < 1e-6


def test_hold_teacher_corrects_position_error() -> None:
    state = sample_hold_states(1, np.random.default_rng(9))[0]
    shifted = HoldState(
        ranges=RangerReading(2.0, 2.0, 2.0, 2.0, 2.0, 0.45),
        x_m=1.0,
        y_m=-1.0,
        z_m=0.25,
        vx_m_s=0.0,
        vy_m_s=0.0,
        vz_m_s=0.0,
        roll_rad=state.roll_rad,
        pitch_rad=state.pitch_rad,
        yaw_rad=0.0,
        gyro_x_rad_s=0.0,
        gyro_y_rad_s=0.0,
        gyro_z_rad_s=0.0,
        target_x_m=0.0,
        target_y_m=0.0,
        target_z_m=0.45,
    )

    command = teacher_hold_command(shifted)

    assert command.vx_m_s < 0.0
    assert command.vy_m_s > 0.0
    assert command.vz_m_s > 0.0


def test_hold_policy_forward_shape_matches_command() -> None:
    model = RangerHoldPolicy(hidden_size=16)
    state = sample_hold_states(1, np.random.default_rng(13))[0]
    output = model(np.asarray([normalize_hold_state(state)], dtype=np.float32))

    assert output.shape == (1, hold_command_array(teacher_hold_command(state)).shape[0])
