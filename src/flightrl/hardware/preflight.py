from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import time
from typing import Mapping

from .config import CrazyflieHardwareConfig
from .errors import HardwareSafetyError
from .telemetry import available_log_variables, build_log_configs


SUPERVISOR_INFO_FLAGS = (
    "Can be armed",
    "Is armed",
    "Is auto armed",
    "Can fly",
    "Is flying",
    "Is tumbled",
    "Is locked",
    "Is crashed",
    "HL control is active",
    "Finished HL trajectory",
    "HL control is disabled",
)
BLOCKING_SUPERVISOR_FLAGS = {"Is tumbled", "Is locked", "Is crashed"}


@dataclass(frozen=True, slots=True)
class PreflightReport:
    ok: bool
    details: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def expected_deck_params(config: CrazyflieHardwareConfig) -> tuple[str, ...]:
    params: list[str] = []
    if config.decks.expect_flow_deck:
        params.append("deck.bcFlow2")
    if config.decks.expect_multiranger:
        params.append("deck.bcMultiranger")
    return tuple(params)


def inspect_decks(scf, config: CrazyflieHardwareConfig) -> PreflightReport:
    details: dict[str, str] = {}
    warnings: list[str] = []
    param = getattr(getattr(scf, "cf", scf), "param", None)
    getter = getattr(param, "get_value", None)
    if getter is None:
        return PreflightReport(
            ok=True,
            warnings=("deck parameters could not be read through this cflib object",),
        )

    ok = True
    for name in expected_deck_params(config):
        try:
            value = str(getter(name))
        except Exception as exc:
            warnings.append(f"{name} unavailable: {exc}")
            ok = False
            continue
        details[name] = value
        if value not in {"1", "True", "true"}:
            warnings.append(f"{name} expected 1/true but got {value}")
            ok = False
    return PreflightReport(ok=ok, details=details, warnings=tuple(warnings))


def inspect_log_variables(scf, config: CrazyflieHardwareConfig) -> PreflightReport:
    cf = getattr(scf, "cf", scf)
    toc = getattr(getattr(getattr(cf, "log", None), "toc", None), "toc", None)
    if not isinstance(toc, dict):
        return PreflightReport(
            ok=True,
            warnings=("log variable TOC could not be read through this cflib object",),
        )

    available = available_log_variables(scf, config.logging.variables)
    missing = tuple(variable for variable in config.logging.variables if variable not in available)
    details = {
        "available_groups": ",".join(sorted(toc.keys())),
        "log_variables_requested": str(len(config.logging.variables)),
        "log_variables_available": str(len(available)),
        "log_variables_missing": str(len(missing)),
    }
    if not available:
        return PreflightReport(
            ok=False,
            details=details,
            warnings=("no configured log variables were found in the Crazyflie TOC",),
        )
    return PreflightReport(
        ok=True,
        details=details,
        warnings=tuple(f"optional log variable missing from TOC: {name}" for name in missing),
    )


def decode_supervisor_info(value: int) -> tuple[str, ...]:
    return tuple(name for index, name in enumerate(SUPERVISOR_INFO_FLAGS) if value & (1 << index))


def blocking_supervisor_flags(value: int) -> tuple[str, ...]:
    flags = decode_supervisor_info(value)
    return tuple(flag for flag in flags if flag in BLOCKING_SUPERVISOR_FLAGS)


def read_supervisor_info(scf, modules, config: CrazyflieHardwareConfig, *, timeout_s: float = 1.0) -> int | None:
    logging = replace(config.logging, variables=("supervisor.info",))
    read_config = replace(config, logging=logging)
    latest: dict[str, float] = {}
    with modules.sync_logger_cls(scf, build_log_configs(modules, read_config)) as logger:
        deadline = time() + timeout_s
        while time() < deadline:
            _timestamp, values, _conf = next(logger)
            latest.update({key: float(value) for key, value in values.items()})
            if "supervisor.info" in latest:
                return int(latest["supervisor.info"])
    return None


def require_supervisor_allows_flight(scf, modules, config: CrazyflieHardwareConfig) -> None:
    info = read_supervisor_info(scf, modules, config)
    if info is None:
        return
    blocking = blocking_supervisor_flags(info)
    if blocking:
        flags = ", ".join(blocking)
        raise HardwareSafetyError(f"Crazyflie supervisor blocks flight: {flags} (supervisor.info={info})")
