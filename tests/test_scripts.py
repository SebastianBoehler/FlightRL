from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_script_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/smoke_test.py", "--config", "configs/tasks/hover.toml", "--steps", "4"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "smoke_test_ok" in result.stdout


def test_crazyflie_bringup_dry_run_demo_runs_without_cflib() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/crazyflie_bringup.py", "--dry-run", "demo"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry_run demo command sequence" in result.stdout
    assert "take_off" in result.stdout


def test_crazyflie_log_dry_run_does_not_record_fake_telemetry() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/crazyflie_log.py", "--dry-run", "--duration-s", "0.1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "no telemetry was recorded" in result.stdout
