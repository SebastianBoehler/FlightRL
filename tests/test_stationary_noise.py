from __future__ import annotations

import csv
import subprocess
import sys

from flightrl.sim2real.noise import summarize_stationary_noise


def test_stationary_noise_summary_accepts_stable_log(tmp_path) -> None:
    path = tmp_path / "stable.csv"
    write_noise_log(path, z_step=0.0001, roll_step=0.001)

    report = summarize_stationary_noise(path, min_duration_s=1.0)

    assert report["summary"]["stationary_noise_ready"] is True
    assert report["signals"]["stateEstimate.z"]["std"] > 0.0


def test_stationary_noise_summary_rejects_motion(tmp_path) -> None:
    path = tmp_path / "moving.csv"
    write_noise_log(path, z_step=0.02, roll_step=0.001)

    report = summarize_stationary_noise(path, min_duration_s=1.0, max_position_span_m=0.08)

    assert report["summary"]["stationary_noise_ready"] is False
    assert "position_motion" in report["summary"]["failures"]


def test_stationary_noise_cli_writes_report(tmp_path) -> None:
    path = tmp_path / "stable.csv"
    output = tmp_path / "summary.json"
    write_noise_log(path, z_step=0.0001, roll_step=0.001)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_stationary_noise.py",
            "--input",
            str(path),
            "--output",
            str(output),
            "--min-duration-s",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "stationary_noise_ready=True" in result.stdout
    assert output.exists()
    assert output.with_suffix(".md").exists()


def write_noise_log(path, *, z_step: float, roll_step: float) -> None:
    fields = [
        "host_time_s",
        "stateEstimate.x",
        "stateEstimate.y",
        "stateEstimate.z",
        "stabilizer.roll",
        "stabilizer.pitch",
        "stabilizer.yaw",
        "acc.x",
        "acc.y",
        "acc.z",
        "gyro.x",
        "gyro.y",
        "gyro.z",
        "range.front",
        "range.back",
        "range.left",
        "range.right",
        "range.up",
        "range.zrange",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx in range(60):
            writer.writerow({field: value_for(field, idx, z_step=z_step, roll_step=roll_step) for field in fields})


def value_for(field: str, idx: int, *, z_step: float, roll_step: float) -> float:
    if field == "host_time_s":
        return idx * 0.05
    if field == "stateEstimate.z":
        return 0.4 + idx * z_step
    if field == "stabilizer.roll":
        return idx * roll_step
    if field.startswith("range."):
        return 1000.0 + (idx % 3)
    return 0.0
