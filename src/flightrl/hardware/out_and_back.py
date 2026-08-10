from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import sleep as default_sleep
from typing import Callable

from .errors import HardwareSafetyError
from .motion import MotionCommanderLike


@dataclass(frozen=True, slots=True)
class OutAndBackFlightPlan:
    default_height_m: float = 0.4
    velocity_m_s: float = 0.1
    distance_m: float = 0.5
    hover_s: float = 1.0
    max_flight_s: float = 25.0

    def __post_init__(self) -> None:
        values = (
            self.default_height_m,
            self.velocity_m_s,
            self.distance_m,
            self.hover_s,
            self.max_flight_s,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or value <= 0.0
            for value in values
        ):
            raise HardwareSafetyError(
                "out-and-back motion limits must be finite and positive"
            )
        if self.nominal_duration_s() > self.max_flight_s:
            raise HardwareSafetyError(
                "out-and-back nominal duration exceeds its flight limit"
            )

    def nominal_duration_s(self) -> float:
        return (
            2.0 * self.default_height_m / self.velocity_m_s
            + 2.0 * self.distance_m / self.velocity_m_s
            + 3.0 * self.hover_s
        )


def execute_out_and_back(
    commander: MotionCommanderLike,
    plan: OutAndBackFlightPlan,
    *,
    sleep: Callable[[float], None] = default_sleep,
    on_phase: Callable[[str], None] = lambda _phase: None,
) -> None:
    landed = False
    leg_s = plan.distance_m / plan.velocity_m_s
    try:
        on_phase("takeoff")
        commander.take_off(plan.default_height_m, plan.velocity_m_s)
        sleep(plan.hover_s)
        on_phase("forward")
        commander.start_linear_motion(plan.velocity_m_s, 0.0, 0.0, 0.0)
        sleep(leg_s)
        commander.stop()
        sleep(plan.hover_s)
        on_phase("backward")
        commander.start_linear_motion(-plan.velocity_m_s, 0.0, 0.0, 0.0)
        sleep(leg_s)
        commander.stop()
        sleep(plan.hover_s)
        on_phase("land")
        commander.land(plan.velocity_m_s)
        landed = True
        on_phase("complete")
    finally:
        if not landed:
            commander.stop()
            commander.land(plan.velocity_m_s)
