from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_measured_disturbance_profile_excludes_stress_outlier(tmp_path: Path) -> None:
    logs = [
        ("low", write_log(tmp_path / "low.csv", peak_speed=0.4, peak_vz=0.01)),
        ("mid", write_log(tmp_path / "mid.csv", peak_speed=0.5, peak_vz=-0.02)),
        ("stress", write_log(tmp_path / "stress.csv", peak_speed=5.0, peak_vz=-0.2)),
    ]
    output = tmp_path / "measured.json"
    stress_output = tmp_path / "stress.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_measured_disturbance_profile.py"),
            *[arg for label, path in logs for arg in ("--live-log", f"{label}:{path}")],
            "--output",
            str(output),
            "--stress-output",
            str(stress_output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["disturbance_profile"]["world_accel_xy_m_s2"] == [0.4, 0.5]
    assert report["disturbance_profile"]["world_accel_z_m_s2"] == [-0.02, 0.01]
    assert report["stress_disturbance_profile"]["world_accel_xy_m_s2"] == [5.0, 5.0]
    assert report["summary"]["stress_logs"] == ["stress"]
    assert "stress | `False` | 5.0000" in output.with_suffix(".md").read_text()
    stress_report = json.loads(stress_output.read_text())
    assert stress_report["disturbance_profile"]["name"] == "raw_live_drift_stress"
    assert stress_report["disturbance_profile"]["world_accel_z_m_s2"] == [-0.2, -0.2]
    assert stress_report["summary"]["stress_logs"] == ["stress"]


def write_log(path: Path, *, peak_speed: float, peak_vz: float) -> Path:
    rows = [
        telemetry_row(host_time_s=0.0, vx=0.0, vz=0.0),
        telemetry_row(host_time_s=1.0, vx=peak_speed, vz=peak_vz),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def telemetry_row(*, host_time_s: float, vx: float, vz: float) -> dict[str, float]:
    return {
        "host_time_s": host_time_s,
        "stateEstimate.x": 0.0,
        "stateEstimate.y": 0.0,
        "stateEstimate.z": 0.5,
        "stateEstimate.vx": vx,
        "stateEstimate.vy": 0.0,
        "stateEstimate.vz": vz,
        "stabilizer.roll": 0.0,
        "stabilizer.pitch": 0.0,
        "range.front": 800.0,
        "range.back": 800.0,
        "range.left": 800.0,
        "range.right": 800.0,
        "range.zrange": 500.0,
        "sys.canfly": 1.0,
        "sys.isTumbled": 0.0,
        "action_thrust": 0.0,
        "action_roll_rate": 0.0,
        "action_pitch_rate": 0.0,
        "action_yaw_rate": 0.0,
        "roll_rate_deg_s": 0.0,
        "pitch_rate_deg_s": 0.0,
        "yaw_rate_deg_s": 0.0,
        "thrust_percent": 49.0,
    }
