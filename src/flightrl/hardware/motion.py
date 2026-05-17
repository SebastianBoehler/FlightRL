from __future__ import annotations

from dataclasses import dataclass
from time import sleep as default_sleep
from typing import Callable, Protocol

from .config import CrazyflieHardwareConfig
from .errors import HardwareSafetyError


class MotionCommanderLike(Protocol):
    def take_off(self, height: float, velocity: float) -> None: ...
    def stop(self) -> None: ...
    def turn_left(self, angle_degrees: float, rate: float) -> None: ...
    def turn_right(self, angle_degrees: float, rate: float) -> None: ...
    def land(self, velocity: float) -> None: ...


@dataclass(frozen=True, slots=True)
class DemoFlightPlan:
    default_height_m: float = 0.3
    velocity_m_s: float = 0.15
    turn_rate_deg_s: float = 45.0
    turn_angle_degrees: float = 20.0
    hover_s: float = 2.0
    max_flight_s: float = 20.0

    @classmethod
    def from_config(cls, config: CrazyflieHardwareConfig, *, confirmed: bool) -> "DemoFlightPlan":
        if config.safety.requires_manual_confirm and not confirmed:
            raise HardwareSafetyError("manual confirmation is required before spinning motors")
        safety = config.safety
        return cls(
            default_height_m=safety.default_height_m,
            velocity_m_s=safety.velocity_m_s,
            turn_rate_deg_s=safety.turn_rate_deg_s,
            turn_angle_degrees=safety.turn_angle_deg,
            hover_s=safety.hover_s,
            max_flight_s=safety.max_flight_s,
        )


def execute_demo_flight(
    commander: MotionCommanderLike,
    plan: DemoFlightPlan,
    *,
    sleep: Callable[[float], None] = default_sleep,
) -> None:
    if plan.hover_s * 3 > plan.max_flight_s:
        raise HardwareSafetyError("demo hover timing exceeds max_flight_s")

    landed = False
    try:
        commander.take_off(height=plan.default_height_m, velocity=plan.velocity_m_s)
        sleep(plan.hover_s)
        commander.stop()
        commander.turn_left(plan.turn_angle_degrees, rate=plan.turn_rate_deg_s)
        commander.stop()
        sleep(plan.hover_s)
        commander.turn_right(plan.turn_angle_degrees, rate=plan.turn_rate_deg_s)
        commander.stop()
        sleep(plan.hover_s)
        commander.land(velocity=plan.velocity_m_s)
        landed = True
    finally:
        if not landed:
            commander.stop()
            commander.land(velocity=plan.velocity_m_s)


def build_motion_commander(scf, modules, config: CrazyflieHardwareConfig):
    return modules.motion_commander_cls(scf, default_height=config.safety.default_height_m)
