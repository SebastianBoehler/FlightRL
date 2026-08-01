from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from time import monotonic
from typing import Mapping

from .config import CrazyflieHardwareConfig
from .errors import HardwareSafetyError
from .telemetry import available_log_variables, build_log_configs, next_log_packet


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
PREARM_REQUIRED_SUPERVISOR_FLAGS = {"Can be armed"}
POSTARM_REQUIRED_SUPERVISOR_FLAGS = {"Is armed", "Can fly"}


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
    if config.decks.expect_ai_deck:
        params.append("deck.bcAI")
    if config.decks.expect_zranger:
        params.append("deck.bcZRanger2")
    return tuple(params)


def inspect_decks(scf, config: CrazyflieHardwareConfig) -> PreflightReport:
    details: dict[str, str] = {}
    warnings: list[str] = []
    param = getattr(getattr(scf, "cf", scf), "param", None)
    getter = getattr(param, "get_value", None)
    if getter is None:
        return PreflightReport(
            ok=False,
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
    if type(value) is not int or value < 0:
        raise ValueError("supervisor.info must be a non-negative integer bitfield")
    return tuple(name for index, name in enumerate(SUPERVISOR_INFO_FLAGS) if value & (1 << index))


def blocking_supervisor_flags(value: int) -> tuple[str, ...]:
    flags = decode_supervisor_info(value)
    return tuple(flag for flag in flags if flag in BLOCKING_SUPERVISOR_FLAGS)


def read_supervisor_info(scf, modules, config: CrazyflieHardwareConfig, *, timeout_s: float = 1.0) -> int | None:
    logging = replace(config.logging, variables=("supervisor.info",))
    read_config = replace(config, logging=logging)
    latest: dict[str, float] = {}
    with modules.sync_logger_cls(scf, build_log_configs(modules, read_config)) as logger:
        deadline = monotonic() + timeout_s
        while (remaining := deadline - monotonic()) > 0.0:
            packet = next_log_packet(logger, timeout_s=remaining)
            if packet is None:
                return None
            _timestamp, values, _conf = packet
            latest.update({key: float(value) for key, value in values.items()})
            if "supervisor.info" in latest:
                raw = latest["supervisor.info"]
                if not isfinite(raw) or raw < 0.0 or raw > 65535.0 or not raw.is_integer():
                    raise HardwareSafetyError(f"invalid supervisor.info value: {raw!r}")
                return int(raw)
    return None


def require_supervisor_allows_flight(scf, modules, config: CrazyflieHardwareConfig) -> None:
    _require_supervisor_state(
        scf,
        modules,
        config,
        required=PREARM_REQUIRED_SUPERVISOR_FLAGS,
        phase="before arming",
    )


def require_expected_decks(scf, config: CrazyflieHardwareConfig) -> None:
    report = inspect_decks(scf, config)
    if report.ok:
        return
    detail = "; ".join(report.warnings) or "configured deck check failed"
    raise HardwareSafetyError(f"Crazyflie deck preflight failed: {detail}")


def require_supervisor_is_armed_and_can_fly(scf, modules, config: CrazyflieHardwareConfig) -> None:
    _require_supervisor_state(
        scf,
        modules,
        config,
        required=POSTARM_REQUIRED_SUPERVISOR_FLAGS,
        phase="after arming",
    )


def _require_supervisor_state(
    scf,
    modules,
    config: CrazyflieHardwareConfig,
    *,
    required: set[str],
    phase: str,
) -> None:
    info = read_supervisor_info(scf, modules, config)
    if info is None:
        raise HardwareSafetyError(f"supervisor.info was not received {phase}")
    flags = set(decode_supervisor_info(info))
    blocking = blocking_supervisor_flags(info)
    if blocking:
        names = ", ".join(blocking)
        raise HardwareSafetyError(f"Crazyflie supervisor blocks flight: {names} (supervisor.info={info})")
    missing = sorted(required - flags)
    if missing:
        names = ", ".join(missing)
        raise HardwareSafetyError(f"Crazyflie supervisor lacks {names} {phase} (supervisor.info={info})")
