from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

import pytest

from flightrl.hardware.errors import HardwareSafetyError
from flightrl.hardware.telemetry import next_log_packet

ROOT = Path(__file__).resolve().parents[1]
HARDWARE_CONFIG = "configs/hardware/crazyflie_2_1_brushless_aideck_flow2.toml"


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
        [
            sys.executable,
            "scripts/crazyflie_bringup.py",
            "--config",
            HARDWARE_CONFIG,
            "--dry-run",
            "demo",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry_run demo command sequence" in result.stdout
    assert "take_off" in result.stdout


def test_crazyflie_log_dry_run_does_not_record_fake_telemetry() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/crazyflie_log.py",
            "--config",
            HARDWARE_CONFIG,
            "--dry-run",
            "--duration-s",
            "0.1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "no telemetry was recorded" in result.stdout


@pytest.mark.parametrize("duration", ["nan", "-1", "601"])
def test_crazyflie_log_rejects_invalid_duration(duration: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/crazyflie_log.py",
            "--config",
            HARDWARE_CONFIG,
            "--dry-run",
            "--duration-s",
            duration,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "telemetry duration" in result.stderr


def test_crazyflie_motor_bench_dry_run_runs_without_cflib(tmp_path: Path) -> None:
    blocked_cflib = tmp_path / "cflib"
    blocked_cflib.mkdir()
    (blocked_cflib / "__init__.py").write_text("raise RuntimeError('cflib import attempted')\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/crazyflie_motor_bench.py",
            "--config",
            HARDWARE_CONFIG,
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "dry_run motor bench" in result.stdout
    assert "m4" in result.stdout


def test_crazyflie_motor_bench_dry_run_accepts_single_motor_low_power() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/crazyflie_motor_bench.py",
            "--config",
            HARDWARE_CONFIG,
            "--dry-run",
            "--motors",
            "3",
            "--powers",
            "2500",
            "4000",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "m3: powers=[2500, 4000]" in result.stdout
    assert "m1:" not in result.stdout


def test_crazyflie_motor_bench_rejects_power_above_repository_safety_envelope() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/crazyflie_motor_bench.py",
            "--config",
            HARDWARE_CONFIG,
            "--dry-run",
            "--powers",
            "32001",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "powers must be integers in [1, 32000]" in result.stderr


def test_bounded_log_read_returns_none_at_natural_timeout() -> None:
    from queue import Queue
    from types import SimpleNamespace

    logger = SimpleNamespace(_queue=Queue(), DISCONNECT_EVENT="DISCONNECT")

    assert next_log_packet(logger, timeout_s=0.001) is None


def test_bounded_log_read_rejects_disconnect() -> None:
    from queue import Queue
    from types import SimpleNamespace

    packets = Queue()
    packets.put("DISCONNECT")
    logger = SimpleNamespace(_queue=packets, DISCONNECT_EVENT="DISCONNECT")

    with pytest.raises(HardwareSafetyError, match="disconnected"):
        next_log_packet(logger, timeout_s=0.01)


def test_room_visualizer_filters_and_writes_plot(tmp_path: Path) -> None:
    log = tmp_path / "room.csv"
    log.write_text(
        "host_time_s,stateEstimate.x,stateEstimate.y,stateEstimate.z,stabilizer.roll,stabilizer.pitch,stabilizer.yaw,"
        "range.front,range.back,range.left,range.right,range.up,range.zrange\n"
        "10,1,2,0.01,0,0,0,1000,32766,32766,32766,32766,10\n"
        "11,1.2,2.1,0.40,0,0,0,1000,32766,32766,32766,32766,400\n"
    )
    output = tmp_path / "room.png"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/visualize_crazyflie_room.py",
            "--input",
            str(log),
            "--output",
            str(output),
            "--min-drone-z-m",
            "0.2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert output.exists()
    assert "trajectory samples" in result.stdout


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
