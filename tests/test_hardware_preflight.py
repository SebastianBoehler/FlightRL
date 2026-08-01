from __future__ import annotations

from types import SimpleNamespace

import pytest

from flightrl.hardware.errors import HardwareSafetyError
from flightrl.hardware.preflight import (
    blocking_supervisor_flags,
    decode_supervisor_info,
    inspect_log_variables,
    require_expected_decks,
    require_supervisor_allows_flight,
    require_supervisor_is_armed_and_can_fly,
)


def test_inspect_log_variables_warns_for_optional_missing_entries() -> None:
    config = SimpleNamespace(logging=SimpleNamespace(variables=("stateEstimate.x", "rpm.m1", "missing.value")))
    scf = SimpleNamespace(
        cf=SimpleNamespace(
            log=SimpleNamespace(
                toc=SimpleNamespace(
                    toc={
                        "stateEstimate": {"x": object()},
                        "rpm": {"m1": object()},
                    }
                )
            )
        )
    )

    report = inspect_log_variables(scf, config)

    assert report.ok is True
    assert report.details["log_variables_requested"] == "3"
    assert report.details["log_variables_available"] == "2"
    assert report.details["log_variables_missing"] == "1"
    assert report.warnings == ("optional log variable missing from TOC: missing.value",)


def test_inspect_log_variables_fails_when_no_requested_entries_exist() -> None:
    config = SimpleNamespace(logging=SimpleNamespace(variables=("missing.value",)))
    scf = SimpleNamespace(cf=SimpleNamespace(log=SimpleNamespace(toc=SimpleNamespace(toc={"stateEstimate": {}}))))

    report = inspect_log_variables(scf, config)

    assert report.ok is False
    assert report.warnings == ("no configured log variables were found in the Crazyflie TOC",)


def test_live_deck_gate_fails_when_parameters_cannot_be_read() -> None:
    config = SimpleNamespace(
        decks=SimpleNamespace(
            expect_flow_deck=True,
            expect_multiranger=True,
            expect_ai_deck=False,
            expect_zranger=False,
        )
    )

    with pytest.raises(HardwareSafetyError, match="deck preflight failed"):
        require_expected_decks(object(), config)


def test_supervisor_decoder_does_not_invent_meanings_for_unknown_bits() -> None:
    assert decode_supervisor_info(1 << 11) == ()
    assert blocking_supervisor_flags(1 << 11) == ()


@pytest.mark.parametrize("info", [None, 0, 1 << 5, 1 << 6, 1 << 7])
def test_prearm_supervisor_gate_fails_closed(monkeypatch, info) -> None:
    monkeypatch.setattr("flightrl.hardware.preflight.read_supervisor_info", lambda *_args, **_kwargs: info)

    with pytest.raises(HardwareSafetyError):
        require_supervisor_allows_flight(object(), object(), object())


def test_prearm_supervisor_gate_accepts_can_be_armed(monkeypatch) -> None:
    monkeypatch.setattr("flightrl.hardware.preflight.read_supervisor_info", lambda *_args, **_kwargs: 1 << 0)

    require_supervisor_allows_flight(object(), object(), object())


@pytest.mark.parametrize("info", [None, 0, 1 << 1, 1 << 3])
def test_postarm_supervisor_gate_requires_armed_and_can_fly(monkeypatch, info) -> None:
    monkeypatch.setattr("flightrl.hardware.preflight.read_supervisor_info", lambda *_args, **_kwargs: info)

    with pytest.raises(HardwareSafetyError):
        require_supervisor_is_armed_and_can_fly(object(), object(), object())


def test_postarm_supervisor_gate_accepts_armed_and_can_fly(monkeypatch) -> None:
    monkeypatch.setattr(
        "flightrl.hardware.preflight.read_supervisor_info",
        lambda *_args, **_kwargs: (1 << 1) | (1 << 3),
    )

    require_supervisor_is_armed_and_can_fly(object(), object(), object())
