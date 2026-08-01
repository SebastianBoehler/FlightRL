from __future__ import annotations

from queue import Queue
from types import SimpleNamespace

import pytest

from flightrl.hardware.errors import HardwareSafetyError
from flightrl.hardware.telemetry import (
    TelemetryCsvWriter,
    TelemetrySample,
    available_log_variables,
    build_log_configs,
    with_available_log_variables,
    with_extra_log_variables,
    write_sync_log,
)


def test_telemetry_csv_writes_replay_friendly_rows(tmp_path) -> None:
    path = tmp_path / "flight.csv"
    writer = TelemetryCsvWriter(path, variables=("stabilizer.roll", "pm.vbat"))
    writer.write_sample(
        TelemetrySample(
            host_time_s=1.25,
            crazyflie_time_ms=50,
            values={"stabilizer.roll": 2.0, "pm.vbat": 3.85},
        )
    )
    writer.close()

    assert path.read_text().splitlines() == [
        "host_time_s,crazyflie_time_ms,stabilizer.roll,pm.vbat",
        "1.250000,50,2.0,3.85",
    ]


def test_sync_log_merges_only_packets_with_the_same_device_timestamp(tmp_path) -> None:
    path = tmp_path / "merged.csv"
    config = SimpleNamespace(logging=SimpleNamespace(variables=("a", "b", "c"), period_ms=50))
    modules = SimpleNamespace(log_config_cls=FakeLogConfig, sync_logger_cls=FakeSyncLogger)

    count = write_sync_log(None, modules, config, path, duration_s=0.01)

    assert count == 2
    lines = path.read_text().splitlines()
    assert lines[0] == "host_time_s,crazyflie_time_ms,a,b,c"
    assert lines[1].endswith(",1.0,2.0,3.0")
    assert lines[2].endswith(",4.0,2.0,3.0")


def test_sync_log_does_not_forward_fill_independently_timed_blocks(tmp_path) -> None:
    path = tmp_path / "unsynchronized.csv"
    config = SimpleNamespace(
        logging=SimpleNamespace(variables=("a", "b", "c"), period_ms=50)
    )
    modules = SimpleNamespace(
        log_config_cls=FakeLogConfig,
        sync_logger_cls=UnsynchronizedFakeSyncLogger,
    )

    count = write_sync_log(None, modules, config, path, duration_s=0.01)

    assert count == 0
    assert path.read_text().splitlines() == [
        "host_time_s,crazyflie_time_ms,a,b,c"
    ]


def test_log_configs_use_configured_variable_types() -> None:
    config = SimpleNamespace(
        logging=SimpleNamespace(
            variables=("stabilizer.roll", "motor.m1", "motor.m1req"),
            variable_types={"motor.m1": "uint16_t", "motor.m1req": "int32_t"},
            period_ms=50,
        )
    )
    modules = SimpleNamespace(log_config_cls=FakeLogConfig)

    configs = build_log_configs(modules, config)

    assert configs[0].variables == [
        ("stabilizer.roll", None),
        ("motor.m1", "uint16_t"),
        ("motor.m1req", "int32_t"),
    ]


def test_available_log_variables_filters_against_crazyflie_toc() -> None:
    scf = SimpleNamespace(cf=SimpleNamespace(log=SimpleNamespace(toc=SimpleNamespace(toc={"motor": {"m1": object()}}))))

    variables = available_log_variables(scf, ("motor.m1", "motor.m2", "badname"))

    assert variables == ("motor.m1",)


def test_extra_and_available_log_variables_keep_order_and_filter() -> None:
    config = SimpleNamespace(logging=SimpleNamespace(variables=("stateEstimate.x", "pm.vbat"), period_ms=50))
    extended = with_extra_log_variables(config, ("pm.vbat", "motor.m1", "rpm.m1"))
    scf = SimpleNamespace(
        cf=SimpleNamespace(
            log=SimpleNamespace(
                toc=SimpleNamespace(
                    toc={
                        "stateEstimate": {"x": object()},
                        "motor": {"m1": object()},
                    }
                )
            )
        )
    )

    filtered = with_available_log_variables(scf, extended)

    assert extended.logging.variables == ("stateEstimate.x", "pm.vbat", "motor.m1", "rpm.m1")
    assert filtered.logging.variables == ("stateEstimate.x", "motor.m1")


def test_sync_log_fails_when_toc_contains_no_requested_variables(tmp_path) -> None:
    config = SimpleNamespace(
        logging=SimpleNamespace(variables=("stateEstimate.x",), period_ms=50)
    )
    scf = SimpleNamespace(
        cf=SimpleNamespace(log=SimpleNamespace(toc=SimpleNamespace(toc={})))
    )
    modules = SimpleNamespace(
        log_config_cls=FakeLogConfig,
        sync_logger_cls=FakeSyncLogger,
    )

    with pytest.raises(HardwareSafetyError, match="none of the configured"):
        write_sync_log(scf, modules, config, tmp_path / "empty.csv", duration_s=1.0)

    assert not (tmp_path / "empty.csv").exists()


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), 0.0, -1.0, 601.0])
def test_sync_log_rejects_nonfinite_or_unbounded_duration(tmp_path, duration: float) -> None:
    config = SimpleNamespace(logging=SimpleNamespace(variables=("a",), period_ms=50))
    modules = SimpleNamespace(log_config_cls=FakeLogConfig, sync_logger_cls=FakeSyncLogger)

    with pytest.raises(ValueError, match="telemetry duration"):
        write_sync_log(None, modules, config, tmp_path / "bad.csv", duration_s=duration)


class FakeLogConfig:
    def __init__(self, name: str, period_in_ms: int) -> None:
        self.name = name
        self.period_in_ms = period_in_ms
        self.variables = []

    def add_variable(self, variable: str, kind: str | None = None) -> None:
        self.variables.append((variable, kind))


class FakeSyncLogger:
    def __init__(self, _scf, _configs) -> None:
        self._queue = Queue()
        self.DISCONNECT_EVENT = "DISCONNECT"
        for row in (
            (10, {"a": 1.0}, None),
            (10, {"b": 2.0}, None),
            (10, {"c": 3.0}, None),
            (20, {"a": 4.0}, None),
            (20, {"b": 2.0}, None),
            (20, {"c": 3.0}, None),
        ):
            self._queue.put(row)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class UnsynchronizedFakeSyncLogger(FakeSyncLogger):
    def __init__(self, _scf, _configs) -> None:
        self._queue = Queue()
        self.DISCONNECT_EVENT = "DISCONNECT"
        for row in (
            (10, {"a": 1.0}, None),
            (11, {"b": 2.0}, None),
            (12, {"c": 3.0}, None),
            (20, {"a": 4.0}, None),
        ):
            self._queue.put(row)
