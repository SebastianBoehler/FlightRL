from __future__ import annotations

import csv
import subprocess
import sys

from flightrl.sim2real.actuator import fit_motor_calibration, summarize_motor_bench


def test_motor_calibration_accepts_linear_curves(tmp_path) -> None:
    path = tmp_path / "motor.csv"
    write_motor_bench(path, zero_rpm=False)

    report = fit_motor_calibration(path)

    assert report["summary"]["passed"] is True
    assert report["simulator_priors"]["present"] is True
    assert set(report["simulator_priors"]["relative_motor_gains"]) == {"1", "2", "3", "4"}


def test_motor_calibration_rejects_zero_rpm(tmp_path) -> None:
    path = tmp_path / "motor.csv"
    write_motor_bench(path, zero_rpm=True)

    report = fit_motor_calibration(path)

    assert report["summary"]["passed"] is False
    assert "m1_rpm_signal" in report["summary"]["failures"]
    assert report["simulator_priors"]["present"] is False


def test_motor_bench_summary_flags_missing_rpm_signal(tmp_path) -> None:
    path = tmp_path / "motor.csv"
    write_motor_bench(path, zero_rpm=True)

    report = summarize_motor_bench(path, min_powers=3)

    assert report["passed"] is False
    assert report["failures"] == ["rpm_signal"]


def test_motor_calibration_cli_writes_report(tmp_path) -> None:
    path = tmp_path / "motor.csv"
    output = tmp_path / "fit.json"
    write_motor_bench(path, zero_rpm=False)

    result = subprocess.run(
        [sys.executable, "scripts/fit_motor_bench.py", "--input", str(path), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "motor_calibration_passed=True" in result.stdout
    assert output.exists()
    assert output.with_suffix(".md").exists()


def write_motor_bench(path, *, zero_rpm: bool) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["motor", "power", "rpm", "motor_output", "motor_requested", "vbat"])
        writer.writeheader()
        for motor in range(1, 5):
            gain = 0.42 + 0.01 * motor
            for power in [14000, 20000, 26000, 32000]:
                rpm = 0.0 if zero_rpm else gain * power - 2000.0
                writer.writerow(
                    {
                        "motor": motor,
                        "power": power,
                        "rpm": rpm,
                        "motor_output": power,
                        "motor_requested": power,
                        "vbat": 3.9,
                    }
                )
