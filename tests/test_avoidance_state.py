from __future__ import annotations

from flightrl.hardware.avoidance_policy import AvoidanceCommand
from flightrl.hardware.avoidance_state import EscapeHoldState


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
