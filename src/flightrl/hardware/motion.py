from __future__ import annotations

import warnings
from dataclasses import dataclass
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


class HoverCommanderLike(Protocol):
    def send_hover_setpoint(self, vx: float, vy: float, yawrate: float, zdistance: float) -> None: ...


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


def arm_for_flight(supervisor: SupervisorLike, *, sleep: Callable[[float], None] = default_sleep) -> None:
    supervisor.send_arming_request(True)
    sleep(0.5)


def disarm_after_flight(supervisor: SupervisorLike, *, sleep: Callable[[float], None] = default_sleep) -> None:
    supervisor.send_arming_request(False)
    sleep(0.2)


def arm_crazyflie_for_flight(cf: CrazyflieArmLike, *, sleep: Callable[[float], None] = default_sleep) -> None:
    arm_for_flight(cf.supervisor, sleep=sleep)
    if _system_arm_state(cf.param) is False:
        cf.param.set_value("system.arm", "1")
        sleep(0.5)
    if _system_arm_state(cf.param) is False:
        raise HardwareSafetyError("Crazyflie did not arm; system.arm stayed 0")


def disarm_crazyflie_after_flight(cf: CrazyflieArmLike, *, sleep: Callable[[float], None] = default_sleep) -> None:
    try:
        cf.param.set_value("system.arm", "0")
    finally:
        disarm_after_flight(cf.supervisor, sleep=sleep)


def send_hover_setpoint_compat(
    commander: HoverCommanderLike,
    vx_m_s: float,
    vy_m_s: float,
    yawrate_deg_s: float,
    zdistance_m: float,
) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Using legacy TYPE_HOVER_LEGACY.*",
            category=DeprecationWarning,
        )
        commander.send_hover_setpoint(vx_m_s, vy_m_s, yawrate_deg_s, zdistance_m)


def install_legacy_hover_warning_filter() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Using legacy TYPE_HOVER_LEGACY.*",
        category=DeprecationWarning,
        module=r"cflib\.crazyflie\.commander",
    )


def _system_arm_state(param: ParamLike) -> bool | None:
    try:
        value = param.get_value("system.arm")
    except Exception:
        return None
    return str(value).strip().lower() in {"1", "true"}
