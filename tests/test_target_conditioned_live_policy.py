from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

from flightrl.hardware.avoidance_policy import AvoidanceCommand, RangerReading


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("crazyflie_target_conditioned_policy", ROOT / "scripts" / "crazyflie_target_conditioned_policy.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_projection_gate_suppresses_velocity_toward_close_front() -> None:
    command = AvoidanceCommand(vx_m_s=0.2, vy_m_s=0.0, yawrate_deg_s=0.0, zdistance_m=0.5)
    reading = RangerReading(front_m=0.40, back_m=2.0, left_m=2.0, right_m=2.0, up_m=2.0, zrange_m=0.5)

    gated, scale = MODULE.apply_projected_clearance(command, reading, args())

    assert gated.vx_m_s == 0.0
    assert scale == 0.0


def test_projection_gate_preserves_velocity_when_projected_side_is_clear() -> None:
    command = AvoidanceCommand(vx_m_s=0.2, vy_m_s=-0.1, yawrate_deg_s=0.0, zdistance_m=0.5)
    reading = RangerReading(front_m=2.0, back_m=2.0, left_m=0.4, right_m=2.0, up_m=2.0, zrange_m=0.5)

    gated, scale = MODULE.apply_projected_clearance(command, reading, args())

    assert gated.vx_m_s == command.vx_m_s
    assert gated.vy_m_s == command.vy_m_s
    assert scale == 1.0


def test_abort_guard_triggers_on_critical_horizontal_range() -> None:
    reading = RangerReading(front_m=0.08, back_m=2.0, left_m=2.0, right_m=2.0, up_m=2.0, zrange_m=0.5)

    assert MODULE.should_abort_clearance(reading, Namespace(abort_clearance_m=0.10))


def test_abort_guard_ignores_safe_horizontal_range() -> None:
    reading = RangerReading(front_m=0.12, back_m=2.0, left_m=2.0, right_m=2.0, up_m=0.02, zrange_m=0.02)

    assert not MODULE.should_abort_clearance(reading, Namespace(abort_clearance_m=0.10))


def args() -> Namespace:
    return Namespace(projected_stop_m=0.55, projected_clearance_m=1.10)
