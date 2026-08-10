from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Mapping

from .config import CrazyflieHardwareConfig
from .errors import HardwareConfigError, HardwareSafetyError


ALLOWED_TELEMETRY_URIS = (
    "usb://0",
    "radio://0/80/2M/E7E7E7E7E7",
)
PROCESS_SCHEMA = "flightrl.aideck_flow_preflight_process.v1"
MAXIMUM_REPORT_AGE_S = 900.0
REQUIRED_PERIOD_MS = 50
TELEMETRY_LOG_BLOCKS = 1
DECK_CHECK_TIMEOUT_S = 15.0
TELEMETRY_DURATION_S = 6.0
TELEMETRY_TIMEOUT_S = 15.0
CLEANUP_TIMEOUT_S = 3.0
AUDIBLE_CUES = {
    "motion_start": "/System/Library/Sounds/Glass.aiff",
    "success": "/System/Library/Sounds/Hero.aiff",
    "failure": "/System/Library/Sounds/Basso.aiff",
}
REQUIRED_TELEMETRY = (
    "motion.motion",
    "motion.deltaX",
    "motion.deltaY",
    "motion.squal",
    "range.zrange",
)
REQUIRED_DECK_EXPECTATIONS = {
    "expect_ai_deck": True,
    "expect_flow_deck": True,
    "expect_multiranger": False,
    "expect_zranger": True,
}
DECK_PARAMETERS = {
    "deck.bcAI": True,
    "deck.bcFlow2": True,
    "deck.bcMultiranger": False,
    "deck.bcZRanger2": True,
}


def load_fresh_flow_preflight_report(
    path: str | Path,
    *,
    now_s: float,
) -> tuple[dict[str, object], dict[str, object], bytes]:
    source = Path(path).resolve()
    try:
        raw = source.read_bytes()
        report = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise HardwareSafetyError(f"could not read Flow preflight report: {exc}") from exc
    if not isinstance(report, dict):
        raise HardwareSafetyError("Flow preflight report must be a JSON object")
    age_s = validate_flow_preflight_report(report, now_s=now_s)
    evidence = {
        "schema": PROCESS_SCHEMA,
        "source_path": str(source),
        "embedded_name": "flow_preflight_process.json",
        "sha256": sha256(raw).hexdigest(),
        "age_s": age_s,
    }
    return report, evidence, raw


def validate_flow_preflight_report(
    report: Mapping[str, object],
    *,
    now_s: float,
) -> float:
    outcome = report.get("process_outcome")
    started = report.get("started_host_time_s")
    ended = report.get("ended_host_time_s")
    if not _finite_number(started) or not _finite_number(ended):
        raise HardwareSafetyError("Flow preflight report timestamps are invalid")
    age_s = float(now_s) - float(ended)
    if (
        not isfinite(age_s)
        or float(started) > float(ended)
        or not 0.0 <= age_s <= MAXIMUM_REPORT_AGE_S
    ):
        raise HardwareSafetyError("Flow preflight report is stale or future-dated")
    deck_check = outcome.get("deck_check") if isinstance(outcome, Mapping) else None
    telemetry = outcome.get("telemetry") if isinstance(outcome, Mapping) else None
    expected = (
        report.get("schema") == PROCESS_SCHEMA
        and report.get("controls_drone") is False
        and report.get("non_actuating") is True
        and report.get("props_off_required") is True
        and report.get("rigid_support_required") is True
        and report.get("flight_authority") is False
        and report.get("telemetry_uri") in ALLOWED_TELEMETRY_URIS
        and report.get("deck_expectations") == REQUIRED_DECK_EXPECTATIONS
        and tuple(report.get("telemetry_variables", ())) == REQUIRED_TELEMETRY
        and report.get("telemetry_period_ms") == REQUIRED_PERIOD_MS
        and report.get("telemetry_log_blocks") == TELEMETRY_LOG_BLOCKS
        and report.get("deck_check_timeout_s") == DECK_CHECK_TIMEOUT_S
        and report.get("telemetry_duration_s") == TELEMETRY_DURATION_S
        and report.get("telemetry_timeout_s") == TELEMETRY_TIMEOUT_S
        and report.get("cleanup_timeout_s") == CLEANUP_TIMEOUT_S
        and report.get("audible_cues") == AUDIBLE_CUES
        and report.get("audible_end_cue_error") is None
        and isinstance(outcome, Mapping)
        and outcome.get("succeeded") is True
        and _successful_child(deck_check)
        and _successful_child(telemetry)
        and outcome.get("validation_error") is None
        and outcome.get("flow_preflight_passed") is True
        and outcome.get("packet_loss_free") is True
    )
    if not expected:
        raise HardwareSafetyError("Flow preflight report did not pass its exact contract")
    return age_s


def validate_flow_preflight_config(config: CrazyflieHardwareConfig) -> None:
    if config.radio.uri not in ALLOWED_TELEMETRY_URIS:
        raise HardwareConfigError(
            "Flow preflight requires the exact usb://0 or "
            "radio://0/80/2M/E7E7E7E7E7 URI"
        )
    if any(
        getattr(config.decks, name) is not expected
        for name, expected in REQUIRED_DECK_EXPECTATIONS.items()
    ):
        raise HardwareConfigError(
            "Flow preflight requires the exact AI Deck, Flow Deck, Z-ranger, "
            "and no Multiranger configuration"
        )
    if (
        tuple(config.logging.variables) != REQUIRED_TELEMETRY
        or config.logging.period_ms != REQUIRED_PERIOD_MS
    ):
        raise HardwareConfigError(
            "Flow preflight requires the exact five-variable raw Flow/Z-ranger "
            "profile at 50 ms"
        )


def inspect_exact_flow_preflight_stack(scf) -> dict[str, object]:
    cf = getattr(scf, "cf", scf)
    getter = getattr(getattr(cf, "param", None), "get_value", None)
    if getter is None:
        raise HardwareSafetyError("deck parameters are unavailable")
    deck_values: dict[str, bool] = {}
    for name, expected in DECK_PARAMETERS.items():
        try:
            actual = _deck_boolean(getter(name), name)
        except HardwareSafetyError:
            raise
        except Exception as exc:
            raise HardwareSafetyError(f"could not read {name}: {exc}") from exc
        deck_values[name] = actual
        if actual is not expected:
            raise HardwareSafetyError(f"{name} expected {int(expected)} but got {int(actual)}")

    toc = getattr(getattr(getattr(cf, "log", None), "toc", None), "toc", None)
    if not isinstance(toc, Mapping):
        raise HardwareSafetyError("Crazyflie log TOC is unavailable")
    missing = [name for name in REQUIRED_TELEMETRY if not _toc_contains(toc, name)]
    if missing:
        raise HardwareSafetyError(
            "required Flow preflight TOC variables are missing: " + ", ".join(missing)
        )
    return {
        "deck_parameters": deck_values,
        "toc_variables": list(REQUIRED_TELEMETRY),
        "controls_drone": False,
        "non_actuating": True,
    }


def _deck_boolean(value: object, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise HardwareSafetyError(f"{name} returned an invalid boolean value: {value!r}")


def _toc_contains(toc: Mapping[object, object], variable: str) -> bool:
    group, _, name = variable.partition(".")
    entries = toc.get(group)
    return isinstance(entries, Mapping) and name in entries


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(float(value))
    )


def _successful_child(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and type(value.get("returncode")) is int
        and value.get("returncode") == 0
        and value.get("timed_out") is False
    )
