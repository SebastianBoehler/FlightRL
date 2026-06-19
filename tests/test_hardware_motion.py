from __future__ import annotations

import pytest

from flightrl.hardware.config import CrazyflieHardwareConfig
from flightrl.hardware.errors import HardwareSafetyError
from flightrl.hardware.motion import (
    DemoFlightPlan,
    arm_crazyflie_for_flight,
    arm_for_flight,
    disarm_after_flight,
    disarm_crazyflie_after_flight,
    execute_demo_flight,
    reset_crazyflie_estimator,
)


class FakeCommander:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[float, ...]]] = []

    def take_off(self, height: float, velocity: float) -> None:
        self.calls.append(("take_off", (height, velocity)))

    def stop(self) -> None:
        self.calls.append(("stop", ()))

    def turn_left(self, angle: float, rate: float) -> None:
        self.calls.append(("turn_left", (angle, rate)))

    def turn_right(self, angle: float, rate: float) -> None:
        self.calls.append(("turn_right", (angle, rate)))

    def land(self, velocity: float) -> None:
        self.calls.append(("land", (velocity,)))


class FakeSupervisor:
    def __init__(self) -> None:
        self.requests: list[bool] = []

    def send_arming_request(self, do_arm: bool) -> None:
        self.requests.append(do_arm)


class FakeParam:
    def __init__(self, arm_value: str = "0") -> None:
        self.values = {"system.arm": arm_value}
        self.set_calls: list[tuple[str, str]] = []

    def get_value(self, complete_name: str) -> object:
        return self.values[complete_name]

    def set_value(self, complete_name: str, value: str) -> None:
        self.set_calls.append((complete_name, value))
        self.values[complete_name] = value


class FakeCrazyflie:
    def __init__(self, arm_value: str = "0") -> None:
        self.supervisor = FakeSupervisor()
        self.param = FakeParam(arm_value)


def test_demo_sequence_uses_conservative_motion_primitives() -> None:
    commander = FakeCommander()
    plan = DemoFlightPlan(hover_s=0.0, turn_angle_degrees=15.0)

    execute_demo_flight(commander, plan, sleep=lambda _: None)

    assert commander.calls == [
        ("take_off", (0.3, 0.15)),
        ("stop", ()),
        ("turn_left", (15.0, 45.0)),
        ("stop", ()),
        ("turn_right", (15.0, 45.0)),
        ("stop", ()),
        ("land", (0.15,)),
    ]


def test_demo_requires_confirmation_when_config_demands_it() -> None:
    config = CrazyflieHardwareConfig()

    with pytest.raises(HardwareSafetyError, match="manual confirmation"):
        DemoFlightPlan.from_config(config, confirmed=False)


def test_arm_and_disarm_use_supervisor_requests() -> None:
    supervisor = FakeSupervisor()

    arm_for_flight(supervisor, sleep=lambda _: None)
    disarm_after_flight(supervisor, sleep=lambda _: None)

    assert supervisor.requests == [True, False]


def test_arm_crazyflie_falls_back_to_system_arm_param() -> None:
    cf = FakeCrazyflie(arm_value="0")

    arm_crazyflie_for_flight(cf, sleep=lambda _: None)

    assert cf.supervisor.requests == [True]
    assert cf.param.set_calls == [("system.arm", "1")]


def test_disarm_crazyflie_clears_system_arm_param() -> None:
    cf = FakeCrazyflie(arm_value="1")

    disarm_crazyflie_after_flight(cf, sleep=lambda _: None)

    assert cf.param.set_calls == [("system.arm", "0")]
    assert cf.supervisor.requests == [False]


def test_reset_crazyflie_estimator_toggles_kalman_param() -> None:
    cf = FakeCrazyflie()
    cf.param.values["kalman.resetEstimation"] = "0"

    assert reset_crazyflie_estimator(cf, sleep=lambda _: None) is True
    assert cf.param.set_calls == [("kalman.resetEstimation", "1"), ("kalman.resetEstimation", "0")]
