from __future__ import annotations

from flightrl.hardware.avoidance_policy import AvoidanceCommand, RangerReading
from flightrl.hardware.avoidance_state import DirectionHoldState, EscapeHoldState


def test_escape_hold_keeps_direction_during_emergency() -> None:
    state = EscapeHoldState(hold_steps=2)
    first = AvoidanceCommand(vx_m_s=-0.4, vy_m_s=0.0, yawrate_deg_s=0.0, zdistance_m=0.5)
    flip = AvoidanceCommand(vx_m_s=0.4, vy_m_s=0.0, yawrate_deg_s=0.0, zdistance_m=0.5)

    held, active = state.update(first, emergency=True)
    assert held == first
    assert active is False

    held, active = state.update(flip, emergency=True)
    assert held.vx_m_s == -0.4
    assert active is True


def test_escape_hold_resets_outside_emergency() -> None:
    state = EscapeHoldState(hold_steps=2)
    first = AvoidanceCommand(vx_m_s=-0.4, vy_m_s=0.0, yawrate_deg_s=0.0, zdistance_m=0.5)
    other = AvoidanceCommand(vx_m_s=0.2, vy_m_s=0.0, yawrate_deg_s=0.0, zdistance_m=0.5)

    state.update(first, emergency=True)
    held, active = state.update(other, emergency=False)

    assert held == other
    assert active is False


def test_direction_hold_keeps_escape_vector_inside_window() -> None:
    state = DirectionHoldState(hold_s=0.3, min_speed_m_s=0.1)
    reading = RangerReading(1.0, 1.0, 1.0, 1.0, 2.0, 0.5)
    first = AvoidanceCommand(vx_m_s=0.4, vy_m_s=0.0, yawrate_deg_s=0.0, zdistance_m=0.5)
    flip = AvoidanceCommand(vx_m_s=-0.4, vy_m_s=0.0, yawrate_deg_s=0.0, zdistance_m=0.5)

    held, active = state.update(first, now_s=10.0, reading=reading, range_rate=None)
    assert held == first
    assert active is False

    held, active = state.update(flip, now_s=10.1, reading=reading, range_rate=None)
    assert held.vx_m_s == 0.4
    assert active is True


def test_direction_hold_releases_after_window() -> None:
    state = DirectionHoldState(hold_s=0.2, min_speed_m_s=0.1)
    reading = RangerReading(1.0, 1.0, 1.0, 1.0, 2.0, 0.5)
    state.update(AvoidanceCommand(0.4, 0.0, 0.0, 0.5), now_s=1.0, reading=reading, range_rate=None)

    command, active = state.update(AvoidanceCommand(-0.4, 0.0, 0.0, 0.5), now_s=1.3, reading=reading, range_rate=None)

    assert command.vx_m_s == -0.4
    assert active is False


def test_direction_hold_allows_hard_clearance_override() -> None:
    state = DirectionHoldState(hold_s=0.3, min_speed_m_s=0.1, hard_clearance_m=0.12)
    open_reading = RangerReading(1.0, 1.0, 1.0, 1.0, 2.0, 0.5)
    hard_reading = RangerReading(0.08, 1.0, 1.0, 1.0, 2.0, 0.5)
    state.update(AvoidanceCommand(0.4, 0.0, 0.0, 0.5), now_s=1.0, reading=open_reading, range_rate=None)

    command, active = state.update(AvoidanceCommand(-0.4, 0.0, 0.0, 0.5), now_s=1.1, reading=hard_reading, range_rate=None)

    assert command.vx_m_s == -0.4
    assert active is False
