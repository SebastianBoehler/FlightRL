from __future__ import annotations

import json

import pytest

from flightrl.sim2real.blockers import load_hardware_blockers


def test_load_hardware_blockers_merges_file_and_cli(tmp_path) -> None:
    path = tmp_path / "blockers.json"
    path.write_text(json.dumps({"blockers": ["m3_motor_issue", "motor_bench_failed"]}))

    blockers = load_hardware_blockers(path, ["m3_motor_issue", "manual_stop"])

    assert blockers == ["m3_motor_issue", "manual_stop", "motor_bench_failed"]


def test_load_hardware_blockers_accepts_missing_file_with_extra(tmp_path) -> None:
    blockers = load_hardware_blockers(tmp_path / "missing.json", ["manual_stop"])

    assert blockers == ["manual_stop"]


def test_load_hardware_blockers_rejects_malformed_file(tmp_path) -> None:
    path = tmp_path / "blockers.json"
    path.write_text(json.dumps({"blockers": "m3_motor_issue"}))

    with pytest.raises(ValueError, match="list field"):
        load_hardware_blockers(path)
