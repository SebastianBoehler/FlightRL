from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Mapping, Sequence

from .config import CrazyflieHardwareConfig

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
            log_config.add_variable(variable, variable_types.get(variable, "float"))
        configs.append(log_config)
    return configs


def write_sync_log(scf, modules, config: CrazyflieHardwareConfig, output_path: str | Path, duration_s: float) -> int:
    variables = available_log_variables(scf, tuple(config.logging.variables))
    log_configs = build_log_configs(modules, _with_log_variables(config, variables))
    deadline = time() + duration_s
    latest: dict[str, object] = {}
    count = 0
    with TelemetryCsvWriter(output_path, variables=variables) as writer:
        with modules.sync_logger_cls(scf, log_configs) as logger:
            for crazyflie_time_ms, values, _logconf in logger:
                latest.update(values)
                if all(variable in latest for variable in variables):
                    writer.write_sample(TelemetrySample(time(), int(crazyflie_time_ms), latest.copy()))
                    count += 1
                if time() >= deadline:
                    break
    return count


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
