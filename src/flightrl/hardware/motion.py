from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import sleep as default_sleep
from typing import Callable, Protocol

from .config import CrazyflieHardwareConfig
from .errors import HardwareSafetyError


class MotionCommanderLike(Protocol):
    def take_off(self, height: float, velocity: float) -> None: ...
    def stop(self) -> None: ...
    def start_linear_motion(
        self,
        velocity_x_m: float,
        velocity_y_m: float,
        velocity_z_m: float,
        rate_yaw: float = 0.0,
    ) -> None: ...
    def start_turn_left(self, rate: float) -> None: ...
    def turn_left(self, angle_degrees: float, rate: float) -> None: ...
    def turn_right(self, angle_degrees: float, rate: float) -> None: ...
    def land(self, velocity: float) -> None: ...


class SupervisorLike(Protocol):
    def send_arming_request(self, do_arm: bool) -> None: ...


class ParamLike(Protocol):
    def get_value(self, complete_name: str) -> object: ...
    def set_value(self, complete_name: str, value: str) -> None: ...


class CrazyflieArmLike(Protocol):
    supervisor: SupervisorLike
    param: ParamLike


@dataclass(frozen=True, slots=True)
class PatrolFlightPlan:
    default_height_m: float = 0.4
    velocity_m_s: float = 0.1
    turn_rate_deg_s: float = 8.0
    turn_angle_degrees: float = 20.0
    forward_distance_m: float = 0.3
    hover_s: float = 0.5
    max_flight_s: float = 25.0

    def __post_init__(self) -> None:
        positive = (
            self.default_height_m,
            self.velocity_m_s,
            self.turn_rate_deg_s,
            self.turn_angle_degrees,
            self.forward_distance_m,
            self.max_flight_s,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or value <= 0.0
            for value in positive
        ):
            raise HardwareSafetyError("patrol motion limits must be finite and positive")
        if (
            isinstance(self.hover_s, bool)
            or not isinstance(self.hover_s, (int, float))
            or not isfinite(float(self.hover_s))
            or self.hover_s < 0.0
        ):
            raise HardwareSafetyError("patrol hover duration must be finite and nonnegative")
        duration = self.nominal_duration_s()
        if duration > self.max_flight_s:
            raise HardwareSafetyError(
                f"patrol nominal duration {duration:.2f}s exceeds "
                f"max_flight_s={self.max_flight_s:.2f}"
            )

    def nominal_duration_s(self) -> float:
        return (
            2.0 * self.default_height_m / self.velocity_m_s
            + 2.0 * self.forward_distance_m / self.velocity_m_s
            + self.turn_angle_degrees / self.turn_rate_deg_s
            + 4.0 * self.hover_s
        )


@dataclass(frozen=True, slots=True)
class DemoFlightPlan:
    default_height_m: float = 0.3
    velocity_m_s: float = 0.15
    turn_rate_deg_s: float = 45.0
    turn_angle_degrees: float = 20.0
    hover_s: float = 2.0
    max_flight_s: float = 20.0

    def __post_init__(self) -> None:
        positive = {
            "default_height_m": self.default_height_m,
            "velocity_m_s": self.velocity_m_s,
            "turn_rate_deg_s": self.turn_rate_deg_s,
            "turn_angle_degrees": self.turn_angle_degrees,
            "max_flight_s": self.max_flight_s,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or value <= 0.0
            for value in positive.values()
        ):
            raise HardwareSafetyError("demo motion limits must be finite and positive")
        if (
            isinstance(self.hover_s, bool)
            or not isinstance(self.hover_s, (int, float))
            or not isfinite(float(self.hover_s))
            or self.hover_s < 0.0
        ):
            raise HardwareSafetyError("demo hover duration must be finite and nonnegative")
        duration = self.nominal_duration_s()
        if duration > self.max_flight_s:
            raise HardwareSafetyError(
                f"demo nominal duration {duration:.2f}s exceeds max_flight_s={self.max_flight_s:.2f}"
            )

    def nominal_duration_s(self) -> float:
        return (
            2.0 * self.default_height_m / self.velocity_m_s
            + 2.0 * self.turn_angle_degrees / self.turn_rate_deg_s
            + 3.0 * self.hover_s
        )

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


def execute_patrol_flight(
    commander: MotionCommanderLike,
    plan: PatrolFlightPlan,
    *,
    sleep: Callable[[float], None] = default_sleep,
    on_phase: Callable[[str], None] = lambda _phase: None,
) -> None:
    landed = False
    leg_duration_s = plan.forward_distance_m / plan.velocity_m_s
    try:
        on_phase("takeoff")
        commander.take_off(height=plan.default_height_m, velocity=plan.velocity_m_s)
        sleep(plan.hover_s)
        on_phase("forward_1")
        commander.start_linear_motion(plan.velocity_m_s, 0.0, 0.0, 0.0)
        sleep(leg_duration_s)
        commander.stop()
        sleep(plan.hover_s)
        on_phase("turn_left")
        commander.start_turn_left(plan.turn_rate_deg_s)
        sleep(plan.turn_angle_degrees / plan.turn_rate_deg_s)
        commander.stop()
        sleep(plan.hover_s)
        on_phase("forward_2")
        commander.start_linear_motion(plan.velocity_m_s, 0.0, 0.0, 0.0)
        sleep(leg_duration_s)
        commander.stop()
        sleep(plan.hover_s)
        on_phase("land")
        commander.land(velocity=plan.velocity_m_s)
        landed = True
        on_phase("complete")
    finally:
        if not landed:
            commander.stop()
            commander.land(velocity=plan.velocity_m_s)


def build_motion_commander(scf, modules, config: CrazyflieHardwareConfig):
    return modules.motion_commander_cls(scf, default_height=config.safety.default_height_m)


def arm_for_flight(supervisor: SupervisorLike, *, sleep: Callable[[float], None] = default_sleep) -> None:
    supervisor.send_arming_request(True)
    sleep(0.5)


def disarm_after_flight(supervisor: SupervisorLike, *, sleep: Callable[[float], None] = default_sleep) -> None:
    supervisor.send_arming_request(False)
    sleep(0.2)


def arm_crazyflie_for_flight(cf: CrazyflieArmLike, *, sleep: Callable[[float], None] = default_sleep) -> None:
    arm_for_flight(cf.supervisor, sleep=sleep)
    if _has_param(cf.param, "system.arm") and _system_arm_state(cf.param) is False:
        cf.param.set_value("system.arm", "1")
        sleep(0.5)
    if _has_param(cf.param, "system.arm") and _system_arm_state(cf.param) is False:
        raise HardwareSafetyError("Crazyflie did not arm; system.arm stayed 0")


def disarm_crazyflie_after_flight(cf: CrazyflieArmLike, *, sleep: Callable[[float], None] = default_sleep) -> None:
    try:
        try:
            if _has_param(cf.param, "system.arm"):
                cf.param.set_value("system.arm", "0")
        except Exception:
            pass
    finally:
        disarm_after_flight(cf.supervisor, sleep=sleep)


def _system_arm_state(param: ParamLike) -> bool | None:
    try:
        value = param.get_value("system.arm")
    except Exception:
        return None
    return str(value).strip().lower() in {"1", "true"}


def _has_param(param: ParamLike, complete_name: str) -> bool:
    toc = getattr(getattr(param, "toc", None), "toc", None)
    if not isinstance(toc, dict):
        return True
    group, _, name = complete_name.partition(".")
    return bool(group and name and group in toc and name in toc[group])
