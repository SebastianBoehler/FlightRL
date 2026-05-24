from __future__ import annotations

import csv
import subprocess
import sys

from flightrl.sim2real.latency import summarize_latency


def test_latency_summary_accepts_delayed_command_response(tmp_path) -> None:
    path = tmp_path / "latency.csv"
    write_latency_log(path, delay_steps=3)

    report = summarize_latency(path, max_lag_s=0.4, max_median_latency_s=0.25)

    assert report["summary"]["latency_ready"] is True
    assert report["summary"]["accepted_pairs"] >= 1
    assert 0.1 <= report["summary"]["median_latency_s"] <= 0.2


def test_latency_summary_rejects_missing_pairs(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["host_time_s", "vx_m_s"])
        writer.writeheader()
        writer.writerow({"host_time_s": 0.0, "vx_m_s": 0.0})

    report = summarize_latency(path)

    assert report["summary"]["latency_ready"] is False
    assert "missing_pairs" in report["summary"]["failures"]


def test_latency_cli_writes_report(tmp_path) -> None:
    path = tmp_path / "latency.csv"
    output = tmp_path / "latency.json"
    write_latency_log(path, delay_steps=2)

    result = subprocess.run(
        [sys.executable, "scripts/summarize_hardware_latency.py", "--input", str(path), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "latency_ready=True" in result.stdout
    assert output.exists()
    assert output.with_suffix(".md").exists()


def write_latency_log(path, *, delay_steps: int) -> None:
    dt = 0.05
    commands = [1.0 if 20 <= idx < 45 else 0.0 for idx in range(80)]
    x = 0.0
    with path.open("w", newline="") as handle:
        fields = ["host_time_s", "vx_m_s", "stateEstimate.x", "vy_m_s", "stateEstimate.y", "yawrate_deg_s", "stabilizer.yaw"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, command in enumerate(commands):
            delayed = commands[idx - delay_steps] if idx >= delay_steps else 0.0
            x += delayed * dt
            writer.writerow(
                {
                    "host_time_s": idx * dt,
                    "vx_m_s": command,
                    "stateEstimate.x": x,
                    "vy_m_s": 0.0,
                    "stateEstimate.y": 0.0,
                    "yawrate_deg_s": 0.0,
                    "stabilizer.yaw": 0.0,
                }
            )
