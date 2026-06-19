from __future__ import annotations

import csv
import subprocess
import sys

from flightrl.sim2real.hover_yaw_stability import summarize_hover_yaw_logs


FIELDS = [
    "host_time_s",
    "mode",
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
    "yawrate_deg_s",
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.zrange",
    "pm.vbat",
]


def test_hover_yaw_stability_accepts_clean_zero_xy_log(tmp_path) -> None:
    log = tmp_path / "clean.csv"
    write_log(log, xy_step=0.0004, yawrate=15.0, battery_start=3.9)

    report = summarize_hover_yaw_logs([log], stable_after_s=1.0)

    assert report["summary"]["stability_ready"] is True
    assert report["clean_logs"][0]["command"]["max_abs_vx_m_s"] == 0.0
    assert report["clean_logs"][0]["stable"]["yaw_span_deg"] > 40.0


def test_hover_yaw_stability_rejects_drift(tmp_path) -> None:
    log = tmp_path / "drift.csv"
    write_log(log, xy_step=0.004, yawrate=15.0, battery_start=3.9)

    report = summarize_hover_yaw_logs([log], stable_after_s=1.0, max_xy_span_m=0.1)

    assert report["summary"]["stability_ready"] is False
    assert "xy_drift" in report["clean_logs"][0]["failures"]


def test_hover_yaw_stability_cli_writes_report(tmp_path) -> None:
    log = tmp_path / "clean.csv"
    contaminated = tmp_path / "collision.csv"
    output = tmp_path / "hover_yaw.json"
    write_log(log, xy_step=0.0004, yawrate=15.0, battery_start=3.9)
    write_log(contaminated, xy_step=0.01, yawrate=15.0, battery_start=3.8)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_hover_yaw_stability.py",
            "--log",
            str(log),
            "--contaminated-log",
            str(contaminated),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "stability_ready=True" in result.stdout
    assert output.exists()
    assert output.with_suffix(".md").exists()


def write_log(path, *, xy_step: float, yawrate: float, battery_start: float) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for idx in range(220):
            time_s = idx * 0.05
            mode, cmd_yawrate = mode_for(time_s, yawrate)
            writer.writerow(
                {
                    "host_time_s": time_s,
                    "mode": mode,
                    "vx_m_s": 0.0,
                    "vy_m_s": 0.0,
                    "vz_m_s": 0.0,
                    "yawrate_deg_s": cmd_yawrate,
                    "stateEstimate.x": idx * xy_step,
                    "stateEstimate.y": (idx % 8) * xy_step,
                    "stateEstimate.z": 0.5 + (idx % 4) * 0.002,
                    "stabilizer.roll": 0.1,
                    "stabilizer.pitch": -0.1,
                    "stabilizer.yaw": yaw_for(time_s, yawrate),
                    "range.front": 1000.0,
                    "range.back": 1200.0,
                    "range.left": 800.0,
                    "range.right": 700.0,
                    "range.zrange": 500.0,
                    "pm.vbat": battery_start - idx * 0.0002,
                }
            )


def mode_for(time_s: float, yawrate: float) -> tuple[str, float]:
    if time_s < 2.0:
        return "hover_start", 0.0
    if time_s < 5.0:
        return "yaw_pos", yawrate
    if time_s < 8.0:
        return "yaw_neg", -yawrate
    return "hover_end", 0.0


def yaw_for(time_s: float, yawrate: float) -> float:
    if time_s < 2.0:
        return 0.0
    if time_s < 5.0:
        return (time_s - 2.0) * yawrate
    if time_s < 8.0:
        return 45.0 - (time_s - 5.0) * yawrate
    return 0.0
