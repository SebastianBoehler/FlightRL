from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flightrl.hardware.errors import HardwareSafetyError
from flightrl.hardware.telemetry import next_log_packet
from scripts import crazyflie_bringup, crazyflie_log

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


def test_crazyflie_bringup_dry_run_patrol_prints_bounded_forward_sequence() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/crazyflie_bringup.py",
            "--config",
            HARDWARE_CONFIG,
            "--dry-run",
            "patrol",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "take_off height=0.40 velocity=0.10" in result.stdout
    assert result.stdout.count("start_linear_motion x=0.10") == 2
    assert "start_turn_left rate=8.0" in result.stdout
    assert "land velocity=0.10" in result.stdout
    assert "nominal_duration_s=18.50 max_flight_s=25.00" in result.stdout


def test_crazyflie_bringup_dry_run_out_and_back_has_no_turn() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/crazyflie_bringup.py",
            "--config",
            HARDWARE_CONFIG,
            "--dry-run",
            "out-and-back",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "start_linear_motion x=0.10" in result.stdout
    assert "start_linear_motion x=-0.10" in result.stdout
    assert "start_turn" not in result.stdout
    assert "nominal_duration_s=21.00 max_flight_s=25.00" in result.stdout


def test_live_patrol_starts_fresh_telemetry_before_arming(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    class FakeRecorder:
        sample_count = 12

        def __init__(self, _scf, _modules, _config, output: Path) -> None:
            calls.append(("recorder_init", output))

        def start(self) -> None:
            calls.append(("recorder_start",))

        def wait_ready(self, *, timeout_s: float) -> None:
            calls.append(("recorder_ready", timeout_s))

        def require_safe(self, **kwargs) -> None:
            calls.append(("recorder_safe", kwargs))

        def close(self) -> None:
            calls.append(("recorder_close",))

    class FakeCommander:
        def take_off(self, height: float, velocity: float) -> None:
            calls.append(("take_off", height, velocity))

        def start_linear_motion(self, *values: float) -> None:
            calls.append(("linear", *values))

        def start_turn_left(self, rate: float) -> None:
            calls.append(("turn", rate))

        def stop(self) -> None:
            calls.append(("stop",))

        def land(self, velocity: float) -> None:
            calls.append(("land", velocity))

    @contextmanager
    def fake_context(_config, _modules):
        calls.append(("connected",))
        yield SimpleNamespace(cf=SimpleNamespace())

    def fake_watchdog(duration_s: float, **_kwargs) -> None:
        calls.append(("watchdog", duration_s))

    monkeypatch.setattr(crazyflie_bringup, "FlightTelemetryRecorder", FakeRecorder, raising=False)
    monkeypatch.setattr(crazyflie_bringup, "watchdog_sleep", fake_watchdog, raising=False)
    monkeypatch.setattr(
        crazyflie_bringup,
        "validate_instrumented_patrol",
        lambda _path: {
            "instrumented_patrol_passed": True,
            "failed_checks": [],
        },
        raising=False,
    )
    monkeypatch.setattr(crazyflie_bringup, "_flight_output_dir", lambda: tmp_path, raising=False)
    monkeypatch.setattr(crazyflie_bringup, "require_cflib", lambda: object())
    monkeypatch.setattr(crazyflie_bringup, "sync_crazyflie_context", fake_context)
    monkeypatch.setattr(crazyflie_bringup, "require_expected_decks", lambda *_: calls.append(("decks",)))
    monkeypatch.setattr(
        crazyflie_bringup,
        "require_supervisor_allows_flight",
        lambda *_: calls.append(("supervisor_prearm",)),
    )
    monkeypatch.setattr(
        crazyflie_bringup,
        "arm_crazyflie_for_flight",
        lambda *_: calls.append(("arm",)),
    )
    monkeypatch.setattr(
        crazyflie_bringup,
        "require_supervisor_is_armed_and_can_fly",
        lambda *_: calls.append(("supervisor_armed",)),
    )
    monkeypatch.setattr(
        crazyflie_bringup,
        "build_motion_commander",
        lambda *_: FakeCommander(),
    )
    monkeypatch.setattr(
        crazyflie_bringup,
        "disarm_crazyflie_after_flight",
        lambda *_: calls.append(("disarm",)),
    )
    config = crazyflie_bringup.load_hardware_config(ROOT / HARDWARE_CONFIG)

    result = crazyflie_bringup._patrol(config, dry_run=False, confirmed=True)

    assert result == 0
    assert calls.index(("recorder_start",)) < calls.index(("arm",))
    assert calls.index(("recorder_ready", 2.0)) < calls.index(("arm",))
    assert calls.index(("recorder_close",)) > calls.index(("disarm",))
    assert any(call[0] == "watchdog" for call in calls)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["phase"] for event in events] == [
        "takeoff",
        "forward_1",
        "turn_left",
        "forward_2",
        "land",
        "complete",
    ]
    validation = json.loads((tmp_path / "validation.json").read_text())
    assert validation["instrumented_patrol_passed"] is True


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


def test_crazyflie_log_dry_run_accepts_usb_uri_override() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/crazyflie_log.py",
            "--config",
            HARDWARE_CONFIG,
            "--uri",
            "usb://0",
            "--dry-run",
            "--duration-s",
            "0.1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "dry_run log: uri=usb://0" in result.stdout


def test_crazyflie_log_passes_uri_override_to_sync_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    connected_uris: list[str] = []

    @contextmanager
    def fake_context(config, _modules):
        connected_uris.append(config.radio.uri)
        yield SimpleNamespace(cf=SimpleNamespace())

    class FakeConsoleCapture:
        def __init__(self, _cf, _output) -> None:
            pass

        def start(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(crazyflie_log, "require_cflib", lambda: object())
    monkeypatch.setattr(crazyflie_log, "sync_crazyflie_context", fake_context)
    monkeypatch.setattr(crazyflie_log, "CrazyflieConsoleCapture", FakeConsoleCapture)
    monkeypatch.setattr(crazyflie_log, "write_sync_log", lambda *_args: 1)

    result = crazyflie_log.main(
        [
            "--config",
            HARDWARE_CONFIG,
            "--uri",
            "usb://7",
            "--duration-s",
            "0.1",
            "--output",
            str(tmp_path / "telemetry.csv"),
        ]
    )

    assert result == 0
    assert connected_uris == ["usb://7"]


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
