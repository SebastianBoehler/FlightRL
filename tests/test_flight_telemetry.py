from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from flightrl.hardware.errors import HardwareSafetyError


class FakeCallbacks:
    def __init__(self) -> None:
        self.callbacks = []

    def add_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def remove_callback(self, callback) -> None:
        self.callbacks.remove(callback)

    def call(self, *args) -> None:
        for callback in tuple(self.callbacks):
            callback(*args)


class FakeLogConfig:
    last = None
    instances = []
    started_names = []

    def __init__(self, name: str, period_in_ms: int) -> None:
        self.name = name
        self.period_in_ms = period_in_ms
        self.variables = []
        self.data_received_cb = FakeCallbacks()
        self.error_cb = FakeCallbacks()
        self.started = False
        self.stopped = False
        self.deleted = False
        FakeLogConfig.last = self
        FakeLogConfig.instances.append(self)

    def add_variable(self, variable: str, kind: str | None = None) -> None:
        self.variables.append((variable, kind))

    def start(self) -> None:
        self.started = True
        FakeLogConfig.started_names.append(self.name)

    def stop(self) -> None:
        self.stopped = True

    def delete(self) -> None:
        self.deleted = True


class FakeLogSubsystem:
    def __init__(self) -> None:
        self.configs = []
        self.toc = SimpleNamespace(
            toc={
                "stateEstimate": {
                    name: object()
                    for name in ("x", "y", "z", "roll", "pitch", "yaw")
                },
                "pm": {"vbat": object(), "state": object()},
                "stabilizer": {
                    name: object()
                    for name in ("roll", "pitch", "yaw")
                },
                "range": {
                    name: object()
                    for name in ("front", "back", "left", "right", "up", "zrange")
                },
                "motion": {"motion": object(), "squal": object()},
            }
        )

    def add_config(self, config) -> None:
        self.configs.append(config)


def flight_telemetry_module():
    try:
        return importlib.import_module("flightrl.hardware.flight_telemetry")
    except ModuleNotFoundError:
        pytest.fail("flight telemetry recorder module is missing")


def fake_context(*, expect_multiranger: bool = False):
    FakeLogConfig.instances = []
    FakeLogConfig.started_names = []
    config = SimpleNamespace(
        decks=SimpleNamespace(expect_multiranger=expect_multiranger),
        logging=SimpleNamespace(
            period_ms=50,
            variables=("unused.value",),
            variable_types={"pm.vbat": "float", "pm.state": "int8_t"},
        )
    )
    scf = SimpleNamespace(cf=SimpleNamespace(log=FakeLogSubsystem()))
    modules = SimpleNamespace(log_config_cls=FakeLogConfig)
    return config, scf, modules


def emit_valid_row(*, battery_v: float = 4.0, power_state: int = 0) -> None:
    log_config = next(
        config for config in FakeLogConfig.instances if config.name == "FlightRLFlight"
    )
    log_config.data_received_cb.call(
        1234,
        {
            "stateEstimate.x": 0.1,
            "stateEstimate.y": 0.2,
            "stateEstimate.z": 0.4,
            "stateEstimate.yaw": 3.0,
            "stateEstimate.roll": 1.25,
            "stateEstimate.pitch": -2.5,
            "pm.vbat": battery_v,
            "pm.state": power_state,
        },
        log_config,
    )


def emit_valid_range_row(*, front_mm: int = 900, zrange_mm: int = 400) -> None:
    log_config = next(
        config for config in FakeLogConfig.instances if config.name == "FlightRLRanger"
    )
    log_config.data_received_cb.call(
        1230,
        {
            "stabilizer.roll": 1.0,
            "stabilizer.pitch": -2.0,
            "stabilizer.yaw": 3.0,
            "range.front": front_mm,
            "range.back": 1000,
            "range.left": 1100,
            "range.right": 1200,
            "range.up": 1300,
            "range.zrange": zrange_mm,
            "motion.motion": 176,
            "motion.squal": 100,
        },
        log_config,
    )


def test_flight_telemetry_records_exact_ready_row_and_closes(tmp_path: Path) -> None:
    module = flight_telemetry_module()
    recorder_type = getattr(module, "FlightTelemetryRecorder", None)
    assert recorder_type is not None, "flight telemetry recorder is missing"
    config, scf, modules = fake_context()
    output = tmp_path / "flight.csv"
    recorder = recorder_type(scf, modules, config, output)

    recorder.start()
    emit_valid_row()
    recorder.wait_ready(timeout_s=0.1)
    recorder.require_safe(maximum_age_s=1.0)
    recorder.close()

    lines = output.read_text().splitlines()
    assert lines[0] == (
        "host_time_s,crazyflie_time_ms,stateEstimate.x,stateEstimate.y,"
        "stateEstimate.z,stateEstimate.yaw,pm.vbat,pm.state,"
        "stateEstimate.roll,stateEstimate.pitch"
    )
    assert len(lines) == 2
    assert lines[1].endswith(",1234,0.1,0.2,0.4,3.0,4.0,0.0,1.25,-2.5")
    assert FakeLogConfig.last.variables[-2:] == [
        ("stateEstimate.roll", "FP16"),
        ("stateEstimate.pitch", "FP16"),
    ]
    assert FakeLogConfig.last.started is True
    assert FakeLogConfig.last.stopped is True
    assert FakeLogConfig.last.deleted is True


def test_multiranger_flight_telemetry_records_pose_ranges_and_flow_quality(
    tmp_path: Path,
) -> None:
    module = flight_telemetry_module()
    config, scf, modules = fake_context(expect_multiranger=True)
    output = tmp_path / "flight.csv"
    recorder = module.FlightTelemetryRecorder(scf, modules, config, output)

    recorder.start()
    emit_valid_range_row()
    emit_valid_row()
    recorder.wait_ready(timeout_s=0.1)
    recorder.require_safe(maximum_age_s=1.0)
    recorder.close()

    lines = output.read_text().splitlines()
    assert lines[0] == (
        "host_time_s,crazyflie_time_ms,stateEstimate.x,stateEstimate.y,"
        "stateEstimate.z,stateEstimate.yaw,pm.vbat,pm.state,"
        "stateEstimate.roll,stateEstimate.pitch,stabilizer.roll,"
        "stabilizer.pitch,stabilizer.yaw,range.front,range.back,range.left,"
        "range.right,range.up,range.zrange,motion.motion,motion.squal"
    )
    assert len(lines) == 2
    assert lines[1].endswith(
        ",0.1,0.2,0.4,3.0,4.0,0.0,1.25,-2.5,1.0,-2.0,3.0,"
        "900.0,1000.0,1100.0,1200.0,1300.0,400.0,176.0,100.0"
    )
    assert FakeLogConfig.started_names == [
        "FlightRLRanger",
        "FlightRLFlight",
    ]
    assert all(item.started and item.stopped and item.deleted for item in FakeLogConfig.instances)


def test_multiranger_flight_watchdog_rejects_close_horizontal_obstacle(
    tmp_path: Path,
) -> None:
    module = flight_telemetry_module()
    config, scf, modules = fake_context(expect_multiranger=True)
    recorder = module.FlightTelemetryRecorder(
        scf,
        modules,
        config,
        tmp_path / "flight.csv",
    )
    recorder.start()
    emit_valid_range_row(front_mm=150)
    emit_valid_row()

    with pytest.raises(HardwareSafetyError, match="range.front.*0.150m"):
        recorder.require_safe(maximum_age_s=1.0)
    recorder.close()


def test_multiranger_flight_watchdog_ignores_ground_plane_before_takeoff(
    tmp_path: Path,
) -> None:
    module = flight_telemetry_module()
    config, scf, modules = fake_context(expect_multiranger=True)
    recorder = module.FlightTelemetryRecorder(
        scf,
        modules,
        config,
        tmp_path / "flight.csv",
    )
    recorder.start()
    emit_valid_range_row(front_mm=150, zrange_mm=20)
    emit_valid_row()

    recorder.require_safe(maximum_age_s=1.0)
    recorder.close()


def test_flight_telemetry_watchdog_allows_voltage_sag_in_firmware_battery_state(tmp_path: Path) -> None:
    module = flight_telemetry_module()
    recorder_type = getattr(module, "FlightTelemetryRecorder", None)
    assert recorder_type is not None, "flight telemetry recorder is missing"
    config, scf, modules = fake_context()
    recorder = recorder_type(scf, modules, config, tmp_path / "flight.csv")

    recorder.start()
    emit_valid_row(battery_v=3.6, power_state=0)

    recorder.require_safe(maximum_age_s=1.0)
    recorder.close()


def test_flight_telemetry_watchdog_rejects_firmware_low_power_state(tmp_path: Path) -> None:
    module = flight_telemetry_module()
    recorder_type = getattr(module, "FlightTelemetryRecorder", None)
    config, scf, modules = fake_context()
    recorder = recorder_type(scf, modules, config, tmp_path / "flight.csv")
    recorder.start()
    emit_valid_row(battery_v=3.3, power_state=3)

    with pytest.raises(HardwareSafetyError, match="low-power"):
        recorder.require_safe(maximum_age_s=1.0)
    recorder.close()


def test_watchdog_sleep_checks_telemetry_throughout_motion_segment() -> None:
    module = flight_telemetry_module()
    watchdog_sleep = getattr(module, "watchdog_sleep", None)
    assert watchdog_sleep is not None, "flight watchdog sleep is missing"

    class FakeRecorder:
        def __init__(self) -> None:
            self.checks = 0

        def require_safe(self, **_kwargs) -> None:
            self.checks += 1

    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0
            self.sleeps = []

        def monotonic(self) -> float:
            return self.now

        def sleep(self, duration_s: float) -> None:
            self.sleeps.append(duration_s)
            self.now += duration_s

    recorder = FakeRecorder()
    clock = FakeClock()

    watchdog_sleep(
        0.12,
        recorder=recorder,
        maximum_age_s=0.25,
        poll_interval_s=0.05,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert recorder.checks == 3
    assert clock.sleeps == pytest.approx([0.05, 0.05, 0.02])
