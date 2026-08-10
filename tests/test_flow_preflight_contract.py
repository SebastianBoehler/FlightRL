from __future__ import annotations

import importlib
import json

import pytest

from flightrl.hardware.errors import HardwareSafetyError
from flow_preflight_test_support import passing_flow_preflight_report


def test_fresh_preflight_report_requires_successful_bounded_children(tmp_path) -> None:
    contract = importlib.import_module("flightrl.hardware.flow_preflight_contract")
    report = passing_flow_preflight_report(started_s=980.0, ended_s=990.0)
    report["process_outcome"]["deck_check"]["timed_out"] = True
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(report))

    with pytest.raises(HardwareSafetyError, match="exact contract"):
        contract.load_fresh_flow_preflight_report(path, now_s=1000.0)


def test_fresh_preflight_report_accepts_exact_bounded_contract(tmp_path) -> None:
    contract = importlib.import_module("flightrl.hardware.flow_preflight_contract")
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(passing_flow_preflight_report(started_s=980.0, ended_s=990.0))
    )

    report, evidence, raw = contract.load_fresh_flow_preflight_report(
        path, now_s=1000.0
    )

    assert report["process_outcome"]["succeeded"] is True
    assert evidence["age_s"] == 10.0
    assert len(evidence["sha256"]) == 64
    assert raw == path.read_bytes()


def test_fresh_preflight_report_accepts_exact_bounded_radio_contract(tmp_path) -> None:
    contract = importlib.import_module("flightrl.hardware.flow_preflight_contract")
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(
            passing_flow_preflight_report(
                started_s=980.0,
                ended_s=990.0,
                telemetry_uri="radio://0/80/2M/E7E7E7E7E7",
            )
        )
    )

    report, evidence, _raw = contract.load_fresh_flow_preflight_report(
        path, now_s=1000.0
    )

    assert report["telemetry_uri"] == "radio://0/80/2M/E7E7E7E7E7"
    assert evidence["age_s"] == 10.0


def test_fresh_preflight_report_rejects_other_radio_uri(tmp_path) -> None:
    contract = importlib.import_module("flightrl.hardware.flow_preflight_contract")
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(
            passing_flow_preflight_report(
                started_s=980.0,
                ended_s=990.0,
                telemetry_uri="radio://0/81/2M/E7E7E7E7E7",
            )
        )
    )

    with pytest.raises(HardwareSafetyError, match="exact contract"):
        contract.load_fresh_flow_preflight_report(path, now_s=1000.0)
