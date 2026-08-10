from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from paired_capture_test_support import write_valid_paired_run


ROOT = Path(__file__).resolve().parents[1]


def test_paired_capture_validation_checks_alignment_and_stationary_envelope(tmp_path) -> None:
    module_name = "flightrl.hardware.paired_capture_validation"
    assert importlib.util.find_spec(module_name) is not None
    validation = importlib.import_module(module_name)
    run_dir = write_valid_paired_run(tmp_path)

    report = validation.validate_paired_capture(run_dir)

    assert report["paired_capture_passed"] is True
    assert report["checks"]["camera_telemetry_overlap"] is True
    assert report["metrics"]["camera_rate_hz"] == pytest.approx(65.0)
    assert report["metrics"]["maximum_nearest_telemetry_gap_s"] <= 0.0251
    assert report["training_authority"] is False
    assert report["deployment_authority"] is False
    assert report["shadow_authority"] is False
    assert report["flight_authority"] is False


def test_paired_capture_validation_rejects_nonoverlapping_host_clocks(tmp_path) -> None:
    module_name = "flightrl.hardware.paired_capture_validation"
    assert importlib.util.find_spec(module_name) is not None
    validation = importlib.import_module(module_name)
    run_dir = write_valid_paired_run(tmp_path, telemetry_start_s=200.0)

    report = validation.validate_paired_capture(run_dir)

    assert report["paired_capture_passed"] is False
    assert report["checks"]["camera_telemetry_overlap"] is False


def test_paired_capture_validation_rejects_logged_packet_loss(tmp_path) -> None:
    module_name = "flightrl.hardware.paired_capture_validation"
    assert importlib.util.find_spec(module_name) is not None
    validation = importlib.import_module(module_name)
    run_dir = write_valid_paired_run(tmp_path)
    (run_dir / "telemetry.log").write_text("Too many packets lost\n")

    report = validation.validate_paired_capture(run_dir)

    assert report["paired_capture_passed"] is False
    assert report["checks"]["packet_loss_free"] is False


def test_paired_capture_validation_rejects_average_rate_hiding_camera_stall(
    tmp_path,
) -> None:
    module_name = "flightrl.hardware.paired_capture_validation"
    assert importlib.util.find_spec(module_name) is not None
    validation = importlib.import_module(module_name)
    camera_span_s = 1199.0 / 65.0
    camera_gaps = np.full(1199, (camera_span_s - 0.2) / 1198.0, dtype=np.float64)
    camera_gaps[600] = 0.2
    camera_times = 100.0 + np.concatenate(([0.0], np.cumsum(camera_gaps)))
    run_dir = write_valid_paired_run(tmp_path, camera_times=camera_times)

    report = validation.validate_paired_capture(run_dir)

    assert report["metrics"]["camera_rate_hz"] == pytest.approx(65.0)
    assert report["metrics"]["maximum_camera_host_gap_s"] == pytest.approx(0.2)
    assert report["checks"]["camera_cadence"] is False
    assert report["paired_capture_passed"] is False


def test_paired_capture_validation_rejects_metadata_frame_count_mismatch(tmp_path) -> None:
    validation = importlib.import_module("flightrl.hardware.paired_capture_validation")
    run_dir = write_valid_paired_run(tmp_path)
    capture_path = run_dir / "decoded_frames.npz"
    with np.load(capture_path, allow_pickle=False) as artifact:
        frames = artifact["decoded_frames"]
        times = artifact["host_time_s"]
        metadata = json.loads(str(artifact["metadata_json"]))
    metadata["captured_frames"] = 1199
    np.savez_compressed(
        capture_path,
        decoded_frames=frames,
        host_time_s=times,
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    report = validation.validate_paired_capture(run_dir)

    assert report["checks"]["camera_complete"] is False
    assert report["paired_capture_passed"] is False


def test_paired_capture_validation_rejects_wrong_source_contract(tmp_path) -> None:
    validation = importlib.import_module("flightrl.hardware.paired_capture_validation")
    run_dir = write_valid_paired_run(tmp_path)
    capture_path = run_dir / "decoded_frames.npz"
    with np.load(capture_path, allow_pickle=False) as artifact:
        frames = artifact["decoded_frames"]
        times = artifact["host_time_s"]
        metadata = json.loads(str(artifact["metadata_json"]))
    metadata["source_frame_contract"]["format"] = 0
    np.savez_compressed(
        capture_path,
        decoded_frames=frames,
        host_time_s=times,
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    report = validation.validate_paired_capture(run_dir)

    assert report["checks"]["camera_contract"] is False
    assert report["paired_capture_passed"] is False


def test_paired_capture_validation_rejects_wrong_camera_endpoint(tmp_path) -> None:
    validation = importlib.import_module("flightrl.hardware.paired_capture_validation")
    run_dir = write_valid_paired_run(tmp_path)
    capture_path = run_dir / "decoded_frames.npz"
    with np.load(capture_path, allow_pickle=False) as artifact:
        frames = artifact["decoded_frames"]
        times = artifact["host_time_s"]
        metadata = json.loads(str(artifact["metadata_json"]))
    metadata["configured_source_endpoint"]["host"] = "192.168.4.9"
    np.savez_compressed(
        capture_path,
        decoded_frames=frames,
        host_time_s=times,
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    report = validation.validate_paired_capture(run_dir)

    assert report["checks"]["camera_contract"] is False
    assert report["paired_capture_passed"] is False


def test_paired_capture_validation_rejects_artifacts_outside_process_time(tmp_path) -> None:
    validation = importlib.import_module("flightrl.hardware.paired_capture_validation")
    run_dir = write_valid_paired_run(tmp_path)
    process_path = run_dir / "capture_process.json"
    process = json.loads(process_path.read_text())
    process["started_host_time_s"] = 101.0
    process_path.write_text(json.dumps(process))

    report = validation.validate_paired_capture(run_dir)

    assert report["checks"]["host_time_provenance"] is False
    assert report["paired_capture_passed"] is False


def test_paired_capture_validation_rejects_extra_telemetry_columns(tmp_path) -> None:
    validation = importlib.import_module("flightrl.hardware.paired_capture_validation")
    run_dir = write_valid_paired_run(tmp_path)
    telemetry_path = run_dir / "telemetry.csv"
    rows = telemetry_path.read_text().splitlines()
    rows[0] += ",motion.squal"
    rows[1:] = [f"{row},100" for row in rows[1:]]
    telemetry_path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="exact columns"):
        validation.validate_paired_capture(run_dir)


def test_paired_capture_validation_rejects_tampered_flow_preflight(tmp_path) -> None:
    validation = importlib.import_module("flightrl.hardware.paired_capture_validation")
    run_dir = write_valid_paired_run(tmp_path)
    (run_dir / "flow_preflight_process.json").write_text("{}")

    report = validation.validate_paired_capture(run_dir)

    assert report["checks"]["flow_preflight_contract"] is False
    assert report["paired_capture_passed"] is False


def test_paired_capture_validation_requires_zero_child_returncodes(tmp_path) -> None:
    module_name = "flightrl.hardware.paired_capture_validation"
    assert importlib.util.find_spec(module_name) is not None
    validation = importlib.import_module(module_name)
    run_dir = write_valid_paired_run(tmp_path)
    process_path = run_dir / "capture_process.json"
    process = json.loads(process_path.read_text())
    process["process_outcome"]["camera"]["returncode"] = 1
    process_path.write_text(json.dumps(process))

    report = validation.validate_paired_capture(run_dir)

    assert report["checks"]["process_succeeded"] is False
    assert report["paired_capture_passed"] is False


def test_paired_capture_validation_cli_writes_report(tmp_path) -> None:
    run_dir = write_valid_paired_run(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_aideck_paired_capture.py",
            str(run_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((run_dir / "validation.json").read_text())
    assert report["paired_capture_passed"] is True
