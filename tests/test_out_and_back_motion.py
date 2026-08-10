from __future__ import annotations

import importlib

import pytest


class FakeCommander:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def take_off(self, height: float, velocity: float) -> None:
        self.calls.append(("take_off", height, velocity))

    def start_linear_motion(self, *values: float) -> None:
        self.calls.append(("linear", *values))

    def stop(self) -> None:
        self.calls.append(("stop",))

    def land(self, velocity: float) -> None:
        self.calls.append(("land", velocity))


def test_out_and_back_executes_symmetric_body_frame_legs() -> None:
    module = importlib.import_module("flightrl.hardware.out_and_back")
    plan = module.OutAndBackFlightPlan()
    commander = FakeCommander()
    sleeps: list[float] = []
    phases: list[str] = []

    module.execute_out_and_back(
        commander,
        plan,
        sleep=sleeps.append,
        on_phase=phases.append,
    )

    assert commander.calls == [
        ("take_off", 0.4, 0.1),
        ("linear", 0.1, 0.0, 0.0, 0.0),
        ("stop",),
        ("linear", -0.1, 0.0, 0.0, 0.0),
        ("stop",),
        ("land", 0.1),
    ]
    assert sleeps == pytest.approx([1.0, 5.0, 1.0, 5.0, 1.0])
    assert phases == ["takeoff", "forward", "backward", "land", "complete"]
    assert plan.distance_m == pytest.approx(0.5)
    assert plan.nominal_duration_s() == pytest.approx(21.0)
    assert plan.nominal_duration_s() < plan.max_flight_s
