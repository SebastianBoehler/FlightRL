from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
from time import monotonic
from types import SimpleNamespace

import pytest

from scripts import capture_aideck_with_telemetry as capture_cli
from flow_preflight_test_support import write_fresh_passing_flow_preflight


ROOT = Path(__file__).resolve().parents[1]


def test_bounded_single_process_force_reaps_child_ignoring_termination() -> None:
    paired = importlib.import_module("flightrl.hardware.paired_capture_process")
    child = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)"
    )
    started = monotonic()

    outcome = paired.run_bounded_process(
        command=(sys.executable, "-c", child),
        timeout_s=0.10,
        cleanup_timeout_s=0.10,
    )

    assert monotonic() - started < 1.0
    assert outcome.succeeded is False
    assert outcome.timed_out is True
    assert outcome.returncode == -signal.SIGKILL
    with pytest.raises(ProcessLookupError):
        os.kill(outcome.pid, 0)


def test_bounded_pair_force_reaps_telemetry_that_ignores_termination(tmp_path) -> None:
    module_name = "flightrl.hardware.paired_capture_process"
    assert importlib.util.find_spec(module_name) is not None
    paired = importlib.import_module(module_name)
    ready_path = tmp_path / "telemetry.csv"
    sleeper = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready_path)!r}).write_text('header\\nrow\\n'); "
        "time.sleep(60)"
    )
    started = monotonic()
    outcome = paired.run_bounded_capture_processes(
        camera_command=(sys.executable, "-c", "pass"),
        telemetry_command=(sys.executable, "-c", sleeper),
        telemetry_ready_path=ready_path,
        telemetry_required_columns=("header",),
        telemetry_minimum_values={},
        telemetry_ready_timeout_s=0.50,
        timeout_s=0.30,
        cleanup_timeout_s=0.10,
    )

    assert monotonic() - started < 1.0
    assert outcome.timed_out is True
    assert outcome.camera.returncode == 0
    assert outcome.telemetry.returncode == -signal.SIGKILL
    try:
        os.kill(outcome.telemetry.pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("telemetry process survived bounded cleanup")


def test_bounded_pair_does_not_start_camera_after_telemetry_exits_early(tmp_path) -> None:
    module_name = "flightrl.hardware.paired_capture_process"
    assert importlib.util.find_spec(module_name) is not None
    paired = importlib.import_module(module_name)
    camera_marker = tmp_path / "camera-started"
    camera = (
        "from pathlib import Path; "
        f"Path({str(camera_marker)!r}).write_text('started')"
    )

    with pytest.raises(RuntimeError, match="telemetry process exited before camera start"):
        paired.run_bounded_capture_processes(
            camera_command=(sys.executable, "-c", camera),
            telemetry_command=(sys.executable, "-c", "pass"),
            telemetry_ready_path=tmp_path / "telemetry.csv",
            telemetry_required_columns=("header",),
            telemetry_minimum_values={},
            telemetry_ready_timeout_s=0.10,
            timeout_s=1.0,
            cleanup_timeout_s=0.10,
        )

    assert not camera_marker.exists()


def test_bounded_pair_times_out_before_camera_when_telemetry_never_ready(
    tmp_path,
) -> None:
    module_name = "flightrl.hardware.paired_capture_process"
    paired = importlib.import_module(module_name)
    camera_marker = tmp_path / "camera-started"
    camera = (
        "from pathlib import Path; "
        f"Path({str(camera_marker)!r}).write_text('started')"
    )
    started = monotonic()

    with pytest.raises(TimeoutError, match="first data row"):
        paired.run_bounded_capture_processes(
            camera_command=(sys.executable, "-c", camera),
            telemetry_command=(sys.executable, "-c", "import time; time.sleep(60)"),
            telemetry_ready_path=tmp_path / "telemetry.csv",
            telemetry_required_columns=("header",),
            telemetry_minimum_values={},
            telemetry_ready_timeout_s=0.10,
            timeout_s=1.0,
            cleanup_timeout_s=0.10,
        )

    assert monotonic() - started < 1.0
    assert not camera_marker.exists()


def test_bounded_pair_rejects_partial_telemetry_schema_before_camera(
    tmp_path,
) -> None:
    paired = importlib.import_module("flightrl.hardware.paired_capture_process")
    ready_path = tmp_path / "telemetry.csv"
    camera_marker = tmp_path / "camera-started"
    telemetry = (
        "import time; from pathlib import Path; "
        f"Path({str(ready_path)!r}).write_text('host_time_s\\n1.0\\n'); "
        "time.sleep(60)"
    )
    camera = (
        "from pathlib import Path; "
        f"Path({str(camera_marker)!r}).write_text('started')"
    )

    with pytest.raises(TimeoutError, match="exact telemetry header"):
        paired.run_bounded_capture_processes(
            camera_command=(sys.executable, "-c", camera),
            telemetry_command=(sys.executable, "-c", telemetry),
            telemetry_ready_path=ready_path,
            telemetry_required_columns=("host_time_s", "pm.vbat"),
            telemetry_minimum_values={},
            telemetry_ready_timeout_s=0.10,
            timeout_s=1.0,
            cleanup_timeout_s=0.10,
        )

    assert not camera_marker.exists()


def test_bounded_pair_rejects_low_first_row_battery_before_camera(tmp_path) -> None:
    paired = importlib.import_module("flightrl.hardware.paired_capture_process")
    ready_path = tmp_path / "telemetry.csv"
    camera_marker = tmp_path / "camera-started"
    telemetry = (
        "import time; from pathlib import Path; "
        f"Path({str(ready_path)!r}).write_text('host_time_s,pm.vbat\\n1.0,3.60\\n'); "
        "time.sleep(60)"
    )
    camera = (
        "from pathlib import Path; "
        f"Path({str(camera_marker)!r}).write_text('started')"
    )

    with pytest.raises(RuntimeError, match="pm.vbat.*3.70"):
        paired.run_bounded_capture_processes(
            camera_command=(sys.executable, "-c", camera),
            telemetry_command=(sys.executable, "-c", telemetry),
            telemetry_ready_path=ready_path,
            telemetry_required_columns=("host_time_s", "pm.vbat"),
            telemetry_minimum_values={"pm.vbat": 3.70},
            telemetry_ready_timeout_s=0.50,
            timeout_s=1.0,
            cleanup_timeout_s=0.10,
        )

    assert not camera_marker.exists()


def test_bounded_pair_cues_after_ready_row_and_before_camera(tmp_path) -> None:
    paired = importlib.import_module("flightrl.hardware.paired_capture_process")
    ready_path = tmp_path / "telemetry.csv"
    cue = tmp_path / "camera-start-cue"
    telemetry = (
        "import time; from pathlib import Path; "
        f"Path({str(ready_path)!r}).write_text('host_time_s,pm.vbat\\n1.0,4.05\\n'); "
        "time.sleep(0.2)"
    )
    camera = f"from pathlib import Path; assert Path({str(cue)!r}).is_file()"

    outcome = paired.run_bounded_capture_processes(
        camera_command=(sys.executable, "-c", camera),
        telemetry_command=(sys.executable, "-c", telemetry),
        telemetry_ready_path=ready_path,
        telemetry_required_columns=("host_time_s", "pm.vbat"),
        telemetry_minimum_values={"pm.vbat": 3.70},
        telemetry_ready_timeout_s=0.5,
        timeout_s=1.0,
        cleanup_timeout_s=0.1,
        before_camera=lambda: cue.write_text("start"),
    )

    assert outcome.succeeded is True


def test_capture_cli_persists_process_start_failure_manifest(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "failed-run"
    preflight = write_fresh_passing_flow_preflight(tmp_path)

    def fail_to_start(**_kwargs):
        raise OSError("synthetic process start failure")

    monkeypatch.setattr(capture_cli, "run_bounded_capture_processes", fail_to_start)
    monkeypatch.setattr(capture_cli.shutil, "which", lambda _name: "afplay")
    cues: list[str] = []
    monkeypatch.setattr(capture_cli, "_play_cue", lambda _player, cue: cues.append(cue))

    result = capture_cli.main(
        [
            "--config",
            str(
                ROOT
                / "configs/hardware/crazyflie_2_1_brushless_aideck_flow2_usb_capture.toml"
            ),
            "--run-dir",
            str(run_dir),
            "--flow-preflight-report",
            str(preflight),
        ]
    )

    assert result == 2
    manifest = json.loads((run_dir / "capture_process.json").read_text())
    assert manifest["process_outcome"]["succeeded"] is False
    assert manifest["process_outcome"]["error"]["type"] == "OSError"
    assert cues[-1].endswith("Basso.aiff")


def test_capture_cli_copies_fresh_passing_flow_preflight_into_run(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "paired"
    preflight = write_fresh_passing_flow_preflight(tmp_path)
    process = SimpleNamespace(pid=123, returncode=0)
    outcome = SimpleNamespace(
        succeeded=True,
        timed_out=False,
        elapsed_s=0.1,
        camera=process,
        telemetry=process,
    )
    cues: list[str] = []

    def succeed(**kwargs):
        kwargs["before_camera"]()
        return outcome

    monkeypatch.setattr(capture_cli, "run_bounded_capture_processes", succeed)
    monkeypatch.setattr(capture_cli.shutil, "which", lambda _name: "afplay")
    monkeypatch.setattr(capture_cli, "_play_cue", lambda _player, cue: cues.append(cue))

    result = capture_cli.main(
        [
            "--config",
            str(
                ROOT
                / "configs/hardware/crazyflie_2_1_brushless_aideck_flow2_usb_capture.toml"
            ),
            "--run-dir",
            str(run_dir),
            "--flow-preflight-report",
            str(preflight),
        ]
    )

    assert result == 0
    copied = run_dir / "flow_preflight_process.json"
    assert copied.read_bytes() == preflight.read_bytes()
    manifest = json.loads((run_dir / "capture_process.json").read_text())
    evidence = manifest["flow_preflight_evidence"]
    assert evidence["embedded_name"] == copied.name
    assert len(evidence["sha256"]) == 64
    assert 0.0 <= evidence["age_s"] <= 5.0
    assert cues[0].endswith("Glass.aiff")
    assert cues[-1].endswith("Hero.aiff")
