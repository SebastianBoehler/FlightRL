from __future__ import annotations

import json
from pathlib import Path

from flightrl.sim2real.live_system_id import build_live_system_id_report


HEADER = "host_time_s,vx_m_s,vy_m_s,stateEstimate.vx,stateEstimate.vy,stabilizer.yaw,pm.vbat,pm.batteryLevel,range.front,range.back,range.left,range.right,range.up,range.zrange\n"


def test_live_system_id_estimates_command_tracking_and_profile(tmp_path: Path) -> None:
    log = tmp_path / "flight.csv"
    base_profile = tmp_path / "profile.json"
    rows = []
    for idx in range(80):
        command = 0.6 if idx >= 10 else 0.0
        measured = 0.42 if idx >= 15 else 0.0
        rows.append(f"{idx * 0.01},{command},0,{measured},0,0,3.9,80,500,900,700,800,1200,500")
    log.write_text(HEADER + "\n".join(rows) + "\n")
    base_profile.write_text(json.dumps({"sensor_profile": {"action_lag_s": 0.012, "range_noise_std_m": 0.002}}))

    report = build_live_system_id_report(flight_logs=[log], base_profile=base_profile, name="unit")

    tracking = report["runs"][0]["tracking"]
    assert tracking["samples"] > 20
    assert 0.0 <= tracking["lag_s"] <= 0.35
    assert tracking["gain"] > 0.0
    assert report["sensor_profile"]["name"] == "unit"
    assert report["sensor_profile"]["action_lag_s"] == 0.012
