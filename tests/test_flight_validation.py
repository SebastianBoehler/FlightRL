from __future__ import annotations

import csv
import importlib
import json
from math import cos, radians, sin
from pathlib import Path
import subprocess
import sys

import pytest


PHASE_TIMES = {
    "takeoff": 1.0,
    "forward_1": 5.5,
    "turn_left": 9.0,
    "forward_2": 12.0,
    "land": 15.5,
    "complete": 19.5,
}
ROOT = Path(__file__).resolve().parents[1]


def flight_validation_module():
    try:
        return importlib.import_module("flightrl.hardware.flight_validation")
    except ModuleNotFoundError:
        pytest.fail("flight validation module is missing")


def write_synthetic_patrol(
    root: Path,
    *,
    lateral_drift_m: float = 0.0,
    with_multiranger: bool = False,
    flow_squal: int = 100,
    landed_front_mm: int | None = None,
    host_stall_after_index: int | None = None,
) -> None:
    root.mkdir()
    with (root / "events.jsonl").open("w") as handle:
        for phase, host_time_s in PHASE_TIMES.items():
            handle.write(json.dumps({"host_time_s": host_time_s, "phase": phase}) + "\n")
    with (root / "telemetry.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = (
            "host_time_s",
            "crazyflie_time_ms",
            "stateEstimate.x",
            "stateEstimate.y",
            "stateEstimate.z",
            "stateEstimate.yaw",
            "pm.vbat",
            "pm.state",
            "stateEstimate.roll",
            "stateEstimate.pitch",
        )
        if with_multiranger:
            header += (
                "stabilizer.roll",
                "stabilizer.pitch",
                "stabilizer.yaw",
                "range.front",
                "range.back",
                "range.left",
                "range.right",
                "range.up",
                "range.zrange",
                "motion.motion",
                "motion.squal",
            )
        writer.writerow(header)
        for index in range(391):
            t = index * 0.05
            x = y = yaw = 0.0
            z = min(0.4, max(0.0, (t - 1.0) * 0.1))
            if t >= 5.5:
                fraction = min(1.0, (t - 5.5) / 3.0)
                x = 0.3 * fraction
                y = lateral_drift_m * fraction
            if t >= 9.0:
                yaw = 20.0 * min(1.0, (t - 9.0) / 2.5)
            if t >= 12.0:
                fraction = min(1.0, (t - 12.0) / 3.0)
                x = 0.3 + 0.3 * cos(radians(20.0)) * fraction
                y = lateral_drift_m + 0.3 * sin(radians(20.0)) * fraction
            if t >= 15.5:
                z = max(0.0, 0.4 - (t - 15.5) * 0.1)
            host_time_s = t
            if host_stall_after_index is not None and index >= host_stall_after_index:
                host_time_s += 0.08
            row = (host_time_s, index * 50, x, y, z, yaw, 4.0, 0, 0.0, 0.0)
            if with_multiranger:
                front_mm = (
                    landed_front_mm
                    if landed_front_mm is not None and z < 0.2
                    else 1200
                )
                row += (
                    0.0,
                    0.0,
                    yaw,
                    front_mm,
                    1400,
                    1000,
                    1100,
                    1500,
                    400,
                    176,
                    flow_squal,
                )
            writer.writerow(row)


def test_instrumented_patrol_validation_accepts_straight_complete_run(tmp_path: Path) -> None:
    module = flight_validation_module()
    validate = getattr(module, "validate_instrumented_patrol", None)
    assert validate is not None, "instrumented patrol validator is missing"
    run_dir = tmp_path / "passing"
    write_synthetic_patrol(run_dir)

    report = validate(run_dir)

    assert report["instrumented_patrol_passed"] is True
    assert report["checks"] == {
        "complete_phases": True,
        "forward_1": True,
        "forward_2": True,
        "landed": True,
        "telemetry_cadence": True,
        "telemetry_rows": True,
        "turn_left": True,
        "power_state": True,
    }
    assert report["metrics"]["forward_1"]["forward_displacement_m"] == pytest.approx(0.3)
    assert report["metrics"]["forward_2"]["lateral_displacement_m"] == pytest.approx(0.0)
    assert report["longer_scripted_stage_eligible"] is True
    assert report["flight_authority"] is False


def test_instrumented_patrol_validation_requires_valid_multiranger_map_when_present(
    tmp_path: Path,
) -> None:
    module = flight_validation_module()
    run_dir = tmp_path / "range_calibration"
    write_synthetic_patrol(run_dir, with_multiranger=True)

    report = module.validate_instrumented_patrol(run_dir)

    assert report["instrumented_patrol_passed"] is True
    assert report["range_calibration_passed"] is True
    assert report["checks"]["range_calibration"] is True
    assert report["metrics"]["mapping"]["mapping_ready"] is True
    assert report["metrics"]["mapping"]["active_horizontal_sensors"] == [
        "range.front",
        "range.back",
        "range.left",
        "range.right",
    ]
    assert report["metrics"]["flow"]["quality_ratio"] == 1.0


def test_instrumented_patrol_validation_reports_low_flow_quality(
    tmp_path: Path,
) -> None:
    module = flight_validation_module()
    run_dir = tmp_path / "low_flow_quality"
    write_synthetic_patrol(run_dir, with_multiranger=True, flow_squal=20)

    report = module.validate_instrumented_patrol(run_dir)

    assert report["instrumented_patrol_passed"] is False
    assert report["range_calibration_passed"] is False
    assert report["range_calibration_failed_checks"] == ["flow_quality"]


def test_instrumented_patrol_validation_ignores_landed_ground_plane_returns(
    tmp_path: Path,
) -> None:
    module = flight_validation_module()
    run_dir = tmp_path / "landed_ground_plane"
    write_synthetic_patrol(
        run_dir,
        with_multiranger=True,
        landed_front_mm=150,
    )

    report = module.validate_instrumented_patrol(run_dir)

    assert report["range_calibration_passed"] is True
    assert report["instrumented_patrol_passed"] is True


def test_instrumented_patrol_validation_accepts_host_stall_with_exact_device_cadence(
    tmp_path: Path,
) -> None:
    module = flight_validation_module()
    run_dir = tmp_path / "host_stall"
    write_synthetic_patrol(run_dir, host_stall_after_index=100)

    report = module.validate_instrumented_patrol(run_dir)

    assert report["checks"]["telemetry_cadence"] is True
    assert report["metrics"]["maximum_telemetry_gap_s"] == pytest.approx(0.13)
    assert report["metrics"]["maximum_device_gap_ms"] == 50.0


def test_multiranger_mapping_uses_device_time_when_host_callbacks_burst(
    tmp_path: Path,
) -> None:
    module = flight_validation_module()
    run_dir = tmp_path / "host_callback_burst"
    write_synthetic_patrol(run_dir, with_multiranger=True)
    rows = list(csv.reader((run_dir / "telemetry.csv").open()))
    rows[121][0] = str(float(rows[120][0]) + 0.001)
    with (run_dir / "telemetry.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)

    report = module.validate_instrumented_patrol(run_dir)

    assert report["metrics"]["maximum_device_gap_ms"] == 50.0
    assert report["metrics"]["mapping"]["trajectory_quality"]["speed_glitch_count"] == 0
    assert report["range_calibration_passed"] is True


def test_instrumented_patrol_validation_rejects_lateral_drift(tmp_path: Path) -> None:
    module = flight_validation_module()
    validate = getattr(module, "validate_instrumented_patrol", None)
    assert validate is not None, "instrumented patrol validator is missing"
    run_dir = tmp_path / "drifting"
    write_synthetic_patrol(run_dir, lateral_drift_m=0.12)

    report = validate(run_dir)

    assert report["instrumented_patrol_passed"] is False
    assert report["checks"]["forward_1"] is False
    assert report["longer_scripted_stage_eligible"] is False


def test_instrumented_patrol_validation_accepts_small_negative_landed_estimate(
    tmp_path: Path,
) -> None:
    module = flight_validation_module()
    run_dir = tmp_path / "negative_landed_z"
    write_synthetic_patrol(run_dir)
    rows = list(csv.reader((run_dir / "telemetry.csv").open()))
    rows[-1][4] = "-0.01"
    with (run_dir / "telemetry.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)

    report = module.validate_instrumented_patrol(run_dir)

    assert report["checks"]["landed"] is True


def test_instrumented_patrol_validation_cli_persists_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "passing_cli"
    write_synthetic_patrol(run_dir)

    result = subprocess.run(
        [sys.executable, "scripts/validate_instrumented_patrol.py", str(run_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((run_dir / "validation.json").read_text())
    assert report["instrumented_patrol_passed"] is True
