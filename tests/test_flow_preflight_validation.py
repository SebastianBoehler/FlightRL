from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    "host_time_s,crazyflie_time_ms,motion.motion,motion.deltaX,"
    "motion.deltaY,motion.squal,range.zrange"
)


def _write_flow_log(path: Path, *, healthy: bool) -> None:
    rows = [HEADER]
    for index in range(101):
        moving = 20 <= index < 40
        motion_status = 0xB0 if healthy else 0x30
        delta_x = 3 if healthy and moving else 0
        delta_y = -2 if healthy and moving else 0
        squal = 104 if healthy else 2
        rows.append(
            f"{100.0 + index * 0.05:.6f},{index * 50},{motion_status},"
            f"{delta_x},{delta_y},{squal},310"
        )
    path.write_text("\n".join(rows) + "\n")


def test_flow_preflight_accepts_official_motion_status_with_all_row_quality(
    tmp_path,
) -> None:
    validation = importlib.import_module("flightrl.hardware.flow_preflight_validation")
    log = tmp_path / "flow.csv"
    rows = [
        "host_time_s,crazyflie_time_ms,motion.motion,motion.deltaX,"
        "motion.deltaY,motion.squal,range.zrange"
    ]
    for index in range(101):
        moving = 20 <= index < 40
        rows.append(
            f"{100.0 + index * 0.05:.6f},{index * 50},"
            f"{0xB0},{3 if moving else 0},"
            f"{-2 if moving else 0},104,310"
        )
    log.write_text("\n".join(rows) + "\n")

    report = validation.validate_flow_preflight(log)

    assert report["flow_preflight_passed"] is True
    assert report["checks"]["healthy_status"] is True
    assert report["checks"]["healthy_motion"] is True
    assert report["checks"]["flow_quality"] is True
    assert report["metrics"]["healthy_motion_rows"] == 20
    assert report["metrics"]["healthy_status_quality_ratio"] == 1.0


def test_flow_preflight_requires_healthy_status_motion_and_quality(tmp_path) -> None:
    module_name = "flightrl.hardware.flow_preflight_validation"
    assert importlib.util.find_spec(module_name) is not None
    validation = importlib.import_module(module_name)
    log = tmp_path / "flow.csv"
    _write_flow_log(log, healthy=True)

    report = validation.validate_flow_preflight(log)

    assert report["flow_preflight_passed"] is True
    assert report["checks"]["healthy_motion"] is True
    assert report["metrics"]["healthy_motion_rows"] == 20
    assert report["metrics"]["minimum_healthy_motion_squal"] == 104.0
    assert report["flight_authority"] is False


def test_flow_preflight_rejects_zero_motion_and_low_quality(tmp_path) -> None:
    module_name = "flightrl.hardware.flow_preflight_validation"
    assert importlib.util.find_spec(module_name) is not None
    validation = importlib.import_module(module_name)
    log = tmp_path / "flow.csv"
    _write_flow_log(log, healthy=False)

    report = validation.validate_flow_preflight(log)

    assert report["flow_preflight_passed"] is False
    assert report["checks"]["healthy_motion"] is False
    assert report["checks"]["flow_quality"] is False


def test_flow_preflight_rejects_mostly_low_quality_capture_with_five_good_rows(
    tmp_path,
) -> None:
    validation = importlib.import_module("flightrl.hardware.flow_preflight_validation")
    log = tmp_path / "flow.csv"
    rows = [HEADER]
    for index in range(101):
        healthy = index < 5
        rows.append(
            f"{100.0 + index * 0.05:.6f},{index * 50},"
            f"{0xB0 if healthy else 0x30},{3 if healthy else 0},"
            f"{-2 if healthy else 0},{104 if healthy else 2},310"
        )
    log.write_text("\n".join(rows) + "\n")

    report = validation.validate_flow_preflight(log)

    assert report["flow_preflight_passed"] is False
    assert report["checks"]["healthy_status"] is False
    assert report["checks"]["flow_quality"] is False


def test_flow_preflight_cli_writes_report(tmp_path) -> None:
    log = tmp_path / "flow.csv"
    output = tmp_path / "flow-validation.json"
    _write_flow_log(log, healthy=True)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_aideck_flow_preflight.py",
            str(log),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["flow_preflight_passed"] is True
