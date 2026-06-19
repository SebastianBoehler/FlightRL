from __future__ import annotations

from types import SimpleNamespace

from flightrl.hardware.preflight import inspect_log_variables


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
