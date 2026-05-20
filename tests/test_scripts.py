from __future__ import annotations

import importlib.util
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


def test_crazyflie_room_scan_dry_run_runs_without_cflib() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/crazyflie_room_scan.py", "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry_run scan command" in result.stdout


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


def test_ranger_avoidance_training_and_dry_run_deploy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "avoidance.pt"
    train = subprocess.run(
        [
            sys.executable,
            "scripts/train_ranger_avoidance.py",
            "--samples",
            "128",
            "--epochs",
            "2",
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert checkpoint.exists()
    assert "checkpoint=" in train.stdout

    deploy = subprocess.run(
        [
            sys.executable,
            "scripts/crazyflie_avoidance_policy.py",
            "--checkpoint",
            str(checkpoint),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry_run avoidance command" in deploy.stdout


def test_ranger_hold_training_and_dry_run_deploy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "hold.pt"
    train = subprocess.run(
        [
            sys.executable,
            "scripts/train_ranger_hold.py",
            "--samples",
            "128",
            "--epochs",
            "2",
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert checkpoint.exists()
    assert "checkpoint=" in train.stdout

    output = tmp_path / "hold.csv"
    deploy = subprocess.run(
        [
            sys.executable,
            "scripts/crazyflie_hold_policy.py",
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
    assert "wrote 1 rows" in deploy.stdout


def test_hold_policy_waits_for_complete_telemetry() -> None:
    spec = importlib.util.spec_from_file_location("crazyflie_hold_policy", ROOT / "scripts" / "crazyflie_hold_policy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert not module.has_complete_policy_telemetry({"stateEstimate.x": 0.0})
    assert module.has_complete_policy_telemetry(
        {
            "stabilizer.roll": 0.0,
            "stabilizer.pitch": 0.0,
            "stabilizer.yaw": 0.0,
            "stateEstimate.x": 0.0,
            "stateEstimate.y": 0.0,
            "stateEstimate.z": 0.45,
            "stateEstimate.vx": 0.0,
            "stateEstimate.vy": 0.0,
            "stateEstimate.vz": 0.0,
            "gyro.x": 0.0,
            "gyro.y": 0.0,
            "gyro.z": 0.0,
            "range.front": 1000.0,
            "range.back": 1000.0,
            "range.left": 1000.0,
            "range.right": 1000.0,
            "range.up": 2000.0,
            "range.zrange": 450.0,
            "pm.vbat": 3.8,
        }
    )
