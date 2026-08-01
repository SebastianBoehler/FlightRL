from __future__ import annotations

import json
from pathlib import Path

import pytest

from flightrl.sim2real.live_system_id import build_live_system_id_report


HEADER = "host_time_s,vx_m_s,vy_m_s,stateEstimate.vx,stateEstimate.vy,stabilizer.yaw,pm.vbat,pm.batteryLevel,range.front,range.back,range.left,range.right,range.up,range.zrange\n"


def test_live_system_id_estimates_command_tracking_and_profile(tmp_path: Path) -> None:
    log = tmp_path / "flight.csv"
    base_profile = tmp_path / "profile.json"
    rows = []
    measured = 0.0
    for idx in range(80):
        command = 0.6 if idx >= 10 else 0.0
        delayed_command = 0.6 if idx >= 15 else 0.0
        measured += 0.2 * (0.7 * delayed_command - measured)
        rows.append(f"{idx * 0.01},{command},0,{measured},0,0,3.9,80,500,900,700,800,1200,500")
    log.write_text(HEADER + "\n".join(rows) + "\n")
    base_profile.write_text(json.dumps({"sensor_profile": {"action_lag_s": 0.012, "range_noise_std_m": 0.002}}))

    report = build_live_system_id_report(flight_logs=[log], base_profile=base_profile, name="unit")

    tracking = report["runs"][0]["tracking"]
    assert tracking["samples"] > 20
    assert 0.0 <= tracking["lag_s"] <= 0.35
    assert tracking["gain"] > 0.0
    assert report["summary"]["profile_ready"] is True
    assert report["summary"]["tracking_runs"] == 1
    assert report["summary"]["failures"] == []
    assert report["sensor_profile"]["name"] == "unit"
    assert report["sensor_profile"]["action_lag_s"] == 0.012


def test_live_system_id_does_not_mark_unexcited_log_ready(tmp_path: Path) -> None:
    log = tmp_path / "idle.csv"
    rows = [f"{idx * 0.01},0,0,0,0,0" for idx in range(30)]
    log.write_text(",".join(HEADER.split(",")[:6]) + "\n" + "\n".join(rows) + "\n")

    report = build_live_system_id_report(flight_logs=[log], name="idle")

    assert report["summary"]["profile_ready"] is False
    assert report["summary"]["tracking_runs"] == 0
    assert report["summary"]["tracking_samples"] == 0
    assert report["summary"]["failures"] == ["no_valid_tracking_samples"]
    assert report["response"]["gain"]["median"] is None


def test_live_system_id_does_not_mark_empty_input_ready() -> None:
    report = build_live_system_id_report(flight_logs=[], name="empty")

    assert report["summary"]["profile_ready"] is False
    assert report["summary"]["failures"] == ["no_valid_tracking_samples"]


def test_live_system_id_rejects_missing_required_tracking_columns(tmp_path: Path) -> None:
    log = tmp_path / "host_time_only.csv"
    log.write_text("host_time_s\n0.00\n0.01\n0.02\n")

    with pytest.raises(ValueError, match="missing required tracking columns"):
        build_live_system_id_report(flight_logs=[log], name="invalid")


@pytest.mark.parametrize("malformed", ["", "not-a-number", "nan", "inf"])
def test_live_system_id_rejects_malformed_required_tracking_values(tmp_path: Path, malformed: str) -> None:
    log = tmp_path / "malformed.csv"
    rows = ["0.00,0.2,0,0.1,0,0", f"0.01,0.2,0,{malformed},0,0"]
    log.write_text(",".join(HEADER.split(",")[:6]) + "\n" + "\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="invalid required tracking value for stateEstimate.vx"):
        build_live_system_id_report(flight_logs=[log], name="invalid")
