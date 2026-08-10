from __future__ import annotations

from math import isfinite
from pathlib import Path
from threading import Event, Lock
from time import monotonic as default_monotonic
from time import sleep as default_sleep
from time import time
from typing import Mapping

from .errors import HardwareSafetyError
from .telemetry import (
    TelemetryCsvWriter,
    TelemetrySample,
    available_log_variables,
)


FLIGHT_TELEMETRY_VARIABLES = (
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.yaw",
    "pm.vbat",
    "pm.state",
    "stateEstimate.roll",
    "stateEstimate.pitch",
)
RANGER_FLIGHT_TELEMETRY_VARIABLES = (
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "range.zrange",
    "motion.motion",
    "motion.squal",
)
HORIZONTAL_CLEARANCE_VARIABLES = (
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
)
MINIMUM_CLEARANCE_M = 0.20
GROUNDED_MAXIMUM_ZRANGE_M = 0.10
RANGER_NO_RETURN_MINIMUM_MM = 32766


class FlightTelemetryRecorder:
    def __init__(self, scf, modules, config, output: str | Path) -> None:
        self.scf = scf
        self.modules = modules
        self.config = config
        self.output = Path(output)
        self._ready = Event()
        self._lock = Lock()
        self._writer: TelemetryCsvWriter | None = None
        self._log_config = None
        self._range_log_config = None
        self._range_enabled = bool(
            getattr(getattr(config, "decks", None), "expect_multiranger", False)
        )
        self._latest_host_time_s: float | None = None
        self._latest_values: dict[str, float] | None = None
        self._latest_range_host_time_s: float | None = None
        self._latest_range_values: dict[str, float] | None = None
        self._error: HardwareSafetyError | None = None
        self.sample_count = 0

    def start(self) -> None:
        if self._writer is not None:
            raise HardwareSafetyError("flight telemetry recorder is already running")
        output_variables = FLIGHT_TELEMETRY_VARIABLES
        if self._range_enabled:
            output_variables += RANGER_FLIGHT_TELEMETRY_VARIABLES
        available = available_log_variables(self.scf, output_variables)
        if available != output_variables:
            missing = sorted(set(output_variables) - set(available))
            raise HardwareSafetyError(
                "flight telemetry TOC is missing exact variables: " + ", ".join(missing)
            )
        source_logging = self.config.logging
        variable_types = getattr(source_logging, "variable_types", {})
        log_config = self.modules.log_config_cls(
            name="FlightRLFlight",
            period_in_ms=50,
        )
        for variable in FLIGHT_TELEMETRY_VARIABLES:
            fetch_as = (
                "FP16"
                if variable in {"stateEstimate.roll", "stateEstimate.pitch"}
                else variable_types.get(variable)
            )
            log_config.add_variable(variable, fetch_as)
        range_log_config = None
        if self._range_enabled:
            range_log_config = self.modules.log_config_cls(
                name="FlightRLRanger",
                period_in_ms=50,
            )
            for variable in RANGER_FLIGHT_TELEMETRY_VARIABLES:
                if variable.startswith("stabilizer."):
                    fetch_as = "FP16"
                elif variable.startswith("range."):
                    fetch_as = "uint16_t"
                else:
                    fetch_as = "uint8_t"
                range_log_config.add_variable(variable, fetch_as)
        writer = TelemetryCsvWriter(
            self.output,
            variables=output_variables,
        )
        self._writer = writer
        self._log_config = log_config
        self._range_log_config = range_log_config
        try:
            if range_log_config is not None:
                self.scf.cf.log.add_config(range_log_config)
                range_log_config.data_received_cb.add_callback(self._on_range_data)
                range_log_config.error_cb.add_callback(self._on_error)
                range_log_config.start()
            self.scf.cf.log.add_config(log_config)
            log_config.data_received_cb.add_callback(self._on_data)
            log_config.error_cb.add_callback(self._on_error)
            log_config.start()
        except Exception:
            writer.close()
            self._writer = None
            self._log_config = None
            self._range_log_config = None
            raise

    def wait_ready(self, *, timeout_s: float) -> None:
        timeout = _positive("flight telemetry ready timeout", timeout_s)
        if not self._ready.wait(timeout):
            raise HardwareSafetyError("flight telemetry did not produce a ready row")
        self._raise_error()
        if self._latest_values is None:
            raise HardwareSafetyError("flight telemetry became ready without a valid row")

    def require_safe(
        self,
        *,
        maximum_age_s: float,
    ) -> None:
        maximum_age = _positive("maximum flight telemetry age", maximum_age_s)
        self._raise_error()
        with self._lock:
            host_time_s = self._latest_host_time_s
            values = None if self._latest_values is None else self._latest_values.copy()
        if host_time_s is None or values is None:
            raise HardwareSafetyError("flight telemetry has no valid row")
        age_s = time() - host_time_s
        if not isfinite(age_s) or age_s < 0.0 or age_s > maximum_age:
            raise HardwareSafetyError(
                f"flight telemetry is stale ({age_s:.3f}s > {maximum_age:.3f}s)"
            )
        power_state = values["pm.state"]
        if not power_state.is_integer() or not 0.0 <= power_state <= 4.0:
            raise HardwareSafetyError(f"invalid firmware power state {power_state!r}")
        if int(power_state) == 3:
            raise HardwareSafetyError("firmware reports low-power battery state")
        if int(power_state) == 4:
            raise HardwareSafetyError("firmware reports shutdown power state")
        if self._range_enabled:
            with self._lock:
                range_host_time_s = self._latest_range_host_time_s
            range_age_s = (
                float("inf")
                if range_host_time_s is None
                else time() - range_host_time_s
            )
            if not isfinite(range_age_s) or range_age_s < 0.0 or range_age_s > maximum_age:
                raise HardwareSafetyError(
                    f"Multi-ranger telemetry is stale ({range_age_s:.3f}s > {maximum_age:.3f}s)"
                )
            zrange_mm = values["range.zrange"]
            if not zrange_mm.is_integer() or not 0.0 <= zrange_mm <= 65535.0:
                raise HardwareSafetyError(f"invalid range.zrange value {zrange_mm!r}")
            grounded = zrange_mm < GROUNDED_MAXIMUM_ZRANGE_M * 1000.0
            for variable in HORIZONTAL_CLEARANCE_VARIABLES:
                distance_mm = values[variable]
                if not distance_mm.is_integer() or not 0.0 <= distance_mm <= 65535.0:
                    raise HardwareSafetyError(f"invalid {variable} value {distance_mm!r}")
                if distance_mm >= RANGER_NO_RETURN_MINIMUM_MM:
                    continue
                if grounded and variable != "range.up":
                    continue
                distance_m = distance_mm / 1000.0
                if distance_m < MINIMUM_CLEARANCE_M:
                    raise HardwareSafetyError(
                        f"{variable} clearance is only {distance_m:.3f}m"
                    )

    def close(self) -> None:
        log_config = self._log_config
        range_log_config = self._range_log_config
        writer = self._writer
        if log_config is None or writer is None:
            return
        try:
            log_config.stop()
            log_config.delete()
            log_config.data_received_cb.remove_callback(self._on_data)
            log_config.error_cb.remove_callback(self._on_error)
            if range_log_config is not None:
                range_log_config.stop()
                range_log_config.delete()
                range_log_config.data_received_cb.remove_callback(self._on_range_data)
                range_log_config.error_cb.remove_callback(self._on_error)
        finally:
            with self._lock:
                writer.close()
            self._writer = None
            self._log_config = None
            self._range_log_config = None

    def _on_data(self, timestamp_ms, values, _log_config) -> None:
        try:
            if type(timestamp_ms) is not int or not 0 <= timestamp_ms <= 0xFFFFFFFF:
                raise HardwareSafetyError("flight telemetry timestamp must be uint32")
            if not isinstance(values, Mapping):
                raise HardwareSafetyError("flight telemetry values must be a mapping")
            numeric: dict[str, float] = {}
            for variable in FLIGHT_TELEMETRY_VARIABLES:
                value = float(values[variable])
                if not isfinite(value):
                    raise HardwareSafetyError(
                        f"flight telemetry {variable} must be finite"
                    )
                numeric[variable] = value
            host_time_s = time()
            with self._lock:
                if self._writer is None:
                    return
                if self._range_enabled:
                    if self._latest_range_values is None:
                        return
                    numeric.update(self._latest_range_values)
                self._writer.write_sample(
                    TelemetrySample(host_time_s, timestamp_ms, numeric)
                )
                self._latest_host_time_s = host_time_s
                self._latest_values = numeric
                self.sample_count += 1
            self._ready.set()
        except Exception as exc:
            self._set_error(exc)

    def _on_range_data(self, timestamp_ms, values, _log_config) -> None:
        try:
            if type(timestamp_ms) is not int or not 0 <= timestamp_ms <= 0xFFFFFFFF:
                raise HardwareSafetyError("Multi-ranger telemetry timestamp must be uint32")
            if not isinstance(values, Mapping):
                raise HardwareSafetyError("Multi-ranger telemetry values must be a mapping")
            numeric: dict[str, float] = {}
            for variable in RANGER_FLIGHT_TELEMETRY_VARIABLES:
                value = float(values[variable])
                if not isfinite(value):
                    raise HardwareSafetyError(
                        f"Multi-ranger telemetry {variable} must be finite"
                    )
                numeric[variable] = value
            host_time_s = time()
            with self._lock:
                self._latest_range_host_time_s = host_time_s
                self._latest_range_values = numeric
        except Exception as exc:
            self._set_error(exc)

    def _on_error(self, *args) -> None:
        detail = " ".join(str(value) for value in args)
        self._set_error(HardwareSafetyError(f"flight telemetry log error: {detail}"))

    def _set_error(self, exc: Exception) -> None:
        error = exc if isinstance(exc, HardwareSafetyError) else HardwareSafetyError(str(exc))
        with self._lock:
            self._error = error
        self._ready.set()

    def _raise_error(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise error


def watchdog_sleep(
    duration_s: float,
    *,
    recorder: FlightTelemetryRecorder,
    maximum_age_s: float,
    poll_interval_s: float = 0.05,
    monotonic=default_monotonic,
    sleep=default_sleep,
) -> None:
    duration = _positive("watchdog duration", duration_s)
    poll_interval = _positive("watchdog poll interval", poll_interval_s)
    deadline = monotonic() + duration
    while (remaining := deadline - monotonic()) > 0.0:
        recorder.require_safe(
            maximum_age_s=maximum_age_s,
        )
        sleep(min(poll_interval, remaining))


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number
