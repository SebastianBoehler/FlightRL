from __future__ import annotations

import pytest

from flightrl.hardware.config import CrazyflieHardwareConfig
from flightrl.hardware.errors import HardwareSafetyError
from flightrl.hardware.motion import DemoFlightPlan, arm_for_flight, disarm_after_flight, execute_demo_flight


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
