from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from queue import Empty
from time import monotonic, time
from typing import Mapping, Sequence

from .config import CrazyflieHardwareConfig
from .errors import HardwareSafetyError

MAX_VARIABLES_PER_LOG_BLOCK = 5


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    host_time_s: float
    crazyflie_time_ms: int
    values: Mapping[str, object]


class TelemetryCsvWriter:
    def __init__(self, path: str | Path, *, variables: Sequence[str]) -> None:
        self.path = Path(path)
        self.variables = tuple(variables)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(("host_time_s", "crazyflie_time_ms", *self.variables))

    def write_sample(self, sample: TelemetrySample) -> None:
        self._writer.writerow(
            (
                f"{sample.host_time_s:.6f}",
                sample.crazyflie_time_ms,
                *(sample.values.get(variable, "") for variable in self.variables),
            )
        )

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "TelemetryCsvWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def next_log_packet(logger, *, timeout_s: float):
    """Read one SyncLogger packet with a real wall-clock timeout."""
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise ValueError("log packet timeout must be a finite positive number")
    timeout = float(timeout_s)
    if not isfinite(timeout) or timeout <= 0.0:
        raise ValueError("log packet timeout must be a finite positive number")
    packet_queue = getattr(logger, "_queue", None)
    get = getattr(packet_queue, "get", None)
    if get is None:
        raise HardwareSafetyError("cflib SyncLogger does not expose bounded packet reads")
    try:
        packet = get(timeout=timeout)
    except Empty:
        return None
    if packet == getattr(logger, "DISCONNECT_EVENT", object()):
        raise HardwareSafetyError("Crazyflie disconnected while waiting for telemetry")
    if not isinstance(packet, tuple) or len(packet) != 3:
        raise HardwareSafetyError("cflib returned a malformed telemetry packet")
    return packet


def build_log_configs(modules, config: CrazyflieHardwareConfig):
    configs = []
    variables = tuple(config.logging.variables)
    variable_types = getattr(config.logging, "variable_types", {})
    for index in range(0, len(variables), MAX_VARIABLES_PER_LOG_BLOCK):
        chunk = variables[index : index + MAX_VARIABLES_PER_LOG_BLOCK]
        log_config = modules.log_config_cls(
            name=f"FlightRL{len(configs)}",
            period_in_ms=config.logging.period_ms,
        )
        for variable in chunk:
            fetch_as = variable_types.get(variable)
            if fetch_as is None:
                log_config.add_variable(variable)
            else:
                log_config.add_variable(variable, fetch_as)
        configs.append(log_config)
    return configs


def write_sync_log(scf, modules, config: CrazyflieHardwareConfig, output_path: str | Path, duration_s: float) -> int:
    validate_log_duration(duration_s)
    log_config = with_available_log_variables(scf, config)
    variables = tuple(log_config.logging.variables)
    if not variables:
        raise HardwareSafetyError(
            "none of the configured telemetry variables exist in the Crazyflie TOC"
        )
    log_configs = build_log_configs(modules, log_config)
    if not log_configs:
        raise HardwareSafetyError("no telemetry log blocks were configured")
    deadline = monotonic() + duration_s
    pending_by_timestamp: dict[int, dict[str, object]] = {}
    max_pending_timestamps = max(8, 2 * len(log_configs))
    count = 0
    with TelemetryCsvWriter(output_path, variables=variables) as writer:
        with modules.sync_logger_cls(scf, log_configs) as logger:
            while (remaining := deadline - monotonic()) > 0.0:
                packet = next_log_packet(logger, timeout_s=remaining)
                if packet is None:
                    break
                crazyflie_time_ms, values, _logconf = packet
                if (
                    type(crazyflie_time_ms) is not int
                    or not 0 <= crazyflie_time_ms <= 0xFFFFFFFF
                ):
                    raise HardwareSafetyError(
                        "Crazyflie telemetry timestamp must be uint32 milliseconds"
                    )
                if not isinstance(values, Mapping):
                    raise HardwareSafetyError(
                        "Crazyflie telemetry values must be a mapping"
                    )
                timestamp = crazyflie_time_ms
                pending = pending_by_timestamp.setdefault(timestamp, {})
                pending.update(values)
                if all(variable in pending for variable in variables):
                    writer.write_sample(
                        TelemetrySample(time(), timestamp, pending.copy())
                    )
                    count += 1
                    del pending_by_timestamp[timestamp]
                while len(pending_by_timestamp) > max_pending_timestamps:
                    oldest = next(iter(pending_by_timestamp))
                    del pending_by_timestamp[oldest]
    return count


def validate_log_duration(duration_s: float) -> None:
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
        raise ValueError("telemetry duration must be a finite number in (0, 600]")
    if not isfinite(float(duration_s)) or not 0.0 < duration_s <= 600.0:
        raise ValueError("telemetry duration must be a finite number in (0, 600]")


def available_log_variables(scf, variables: Sequence[str]) -> tuple[str, ...]:
    toc = getattr(getattr(getattr(getattr(scf, "cf", scf), "log", None), "toc", None), "toc", None)
    if not isinstance(toc, dict):
        return tuple(variables)
    available: list[str] = []
    for variable in variables:
        group, _, name = variable.partition(".")
        if group in toc and name in toc[group]:
            available.append(variable)
    return tuple(available)


def with_extra_log_variables(config: CrazyflieHardwareConfig, variables: Sequence[str]):
    return _with_log_variables(config, _unique((*tuple(config.logging.variables), *tuple(variables))))


def with_available_log_variables(scf, config: CrazyflieHardwareConfig):
    return _with_log_variables(config, available_log_variables(scf, tuple(config.logging.variables)))


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _with_log_variables(config: CrazyflieHardwareConfig, variables: tuple[str, ...]):
    from dataclasses import replace
    from dataclasses import is_dataclass
    from types import SimpleNamespace

    if is_dataclass(config) and is_dataclass(config.logging):
        return replace(config, logging=replace(config.logging, variables=variables))
    logging = SimpleNamespace(**vars(config.logging))
    logging.variables = variables
    return SimpleNamespace(**{**vars(config), "logging": logging})


def default_log_path(config: CrazyflieHardwareConfig, *, prefix: str = "flight") -> Path:
    timestamp = time()
    return Path(config.logging.output_dir) / f"{prefix}_{timestamp:.0f}.csv"
