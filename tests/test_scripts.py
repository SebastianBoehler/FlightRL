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


def test_crazyflie_motor_bench_dry_run_runs_without_cflib() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/crazyflie_motor_bench.py", "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry_run motor bench" in result.stdout
    assert "m4" in result.stdout


def test_imitation_hover_training_writes_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_imitation_hover.py",
            "--config",
            "configs/tasks/crazyflie_hover.toml",
            "--updates",
            "1",
            "--steps-per-update",
            "4",
            "--num-envs",
            "8",
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert checkpoint.exists()
    assert "checkpoint=" in result.stdout


def test_policy_monitor_dry_run_writes_predictions(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_imitation_hover.py",
            "--config",
            "configs/tasks/crazyflie_hover.toml",
            "--updates",
            "1",
            "--steps-per-update",
            "2",
            "--num-envs",
            "4",
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = tmp_path / "monitor.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/crazyflie_policy_monitor.py",
            "--checkpoint",
            str(checkpoint),
            "--dry-run",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert output.exists()
    assert "wrote 1 rows" in result.stdout
