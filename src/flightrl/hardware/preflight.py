from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .config import CrazyflieHardwareConfig


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

    missing: list[str] = []
    for variable in config.logging.variables:
        group, _, name = variable.partition(".")
        if not group or not name or group not in toc or name not in toc[group]:
            missing.append(variable)
    if missing:
        return PreflightReport(
            ok=False,
            details={"available_groups": ",".join(sorted(toc.keys()))},
            warnings=tuple(f"log variable missing from TOC: {name}" for name in missing),
        )
    return PreflightReport(ok=True, details={"log_variables": str(len(config.logging.variables))})
