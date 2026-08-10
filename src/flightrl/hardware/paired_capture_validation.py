from __future__ import annotations

import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

import numpy as np

from .errors import HardwareSafetyError
from .flow_preflight_contract import (
    AUDIBLE_CUES,
    MAXIMUM_REPORT_AGE_S,
    PROCESS_SCHEMA as FLOW_PREFLIGHT_SCHEMA,
    validate_flow_preflight_report,
)
from .paired_capture_contract import (
    AIDECK_DECODED_CAPTURE_SCHEMA,
    CLEANUP_TIMEOUT_S,
    MAXIMUM_CAMERA_HOST_GAP_S,
    MAXIMUM_CAMERA_RATE_HZ,
    MINIMUM_CAMERA_RATE_HZ,
    MINIMUM_BATTERY_V,
    OVERALL_TIMEOUT_S,
    PROCESS_SCHEMA,
    REQUIRED_CAMERA_BIND_PORT,
    REQUIRED_CAMERA_FRAMES,
    REQUIRED_CAMERA_SOURCE_ENDPOINT,
    REQUIRED_CAMERA_SOURCE_CONTRACT,
    REQUIRED_TELEMETRY,
    REQUIRED_TELEMETRY_COLUMNS,
    REQUIRED_TELEMETRY_URI,
    TELEMETRY_PERIOD_MS,
    TELEMETRY_DURATION_S,
    TELEMETRY_READY_TIMEOUT_S,
)

SCHEMA = "flightrl.aideck_paired_capture_validation.v1"
PACKET_LOSS_MESSAGE = "too many packets lost"


def validate_paired_capture(run_dir: str | Path) -> dict[str, object]:
    root = Path(run_dir)
    process = json.loads((root / "capture_process.json").read_text())
    log_paths = (root / "camera.log", root / "telemetry.log")
    logs_present = all(path.is_file() for path in log_paths)
    logs = "\n".join(
        path.read_text(errors="replace").lower() for path in log_paths if path.is_file()
    )
    camera = _load_camera(root / "decoded_frames.npz")
    telemetry = _load_telemetry(root / "telemetry.csv")
    frames, camera_times, metadata = camera
    telemetry_times = telemetry["host_time_s"]
    device_times = telemetry["crazyflie_time_ms"]
    process_outcome = process.get("process_outcome", {})
    camera_outcome = process_outcome.get("camera", {})
    telemetry_outcome = process_outcome.get("telemetry", {})

    camera_span = float(camera_times[-1] - camera_times[0])
    camera_rate = (len(camera_times) - 1) / camera_span if camera_span > 0.0 else 0.0
    camera_host_gaps = np.diff(camera_times)
    telemetry_host_gaps = np.diff(telemetry_times)
    telemetry_device_gaps = np.diff(device_times)
    nearest_gaps = _nearest_gaps(camera_times, telemetry_times)
    flow_preflight_contract = _validate_embedded_flow_preflight(root, process)
    host_time_provenance = _validate_host_time_provenance(
        process,
        camera_times,
        telemetry_times,
    )
    spans = {
        name: float(np.ptp(telemetry[name]))
        for name in REQUIRED_TELEMETRY[:-1]
    }
    drop_fraction = float(metadata["dropped_frames"]) / max(1, len(frames))
    checks = {
        "process_contract": process.get("schema") == PROCESS_SCHEMA
        and process.get("non_actuating") is True
        and process.get("telemetry_uri") == REQUIRED_TELEMETRY_URI
        and process.get("telemetry_period_ms") == TELEMETRY_PERIOD_MS
        and process.get("telemetry_duration_s") == TELEMETRY_DURATION_S
        and process.get("telemetry_log_blocks") == 1
        and process.get("camera_requested_frames") == REQUIRED_CAMERA_FRAMES
        and process.get("camera_source_endpoint") == REQUIRED_CAMERA_SOURCE_ENDPOINT
        and process.get("camera_bind_port") == REQUIRED_CAMERA_BIND_PORT
        and tuple(process.get("telemetry_variables", ())) == REQUIRED_TELEMETRY
        and process.get("telemetry_ready_timeout_s") == TELEMETRY_READY_TIMEOUT_S
        and process.get("cleanup_timeout_s") == CLEANUP_TIMEOUT_S
        and process.get("overall_timeout_s") == OVERALL_TIMEOUT_S
        and Path(str(process.get("telemetry_ready_path", ""))).name
        == "telemetry.csv"
        and tuple(process.get("telemetry_ready_columns", ()))
        == REQUIRED_TELEMETRY_COLUMNS
        and process.get("telemetry_ready_minimums")
        == {"pm.vbat": MINIMUM_BATTERY_V}
        and process.get("flow_preflight_required_for_live") is True
        and process.get("audible_cues") == AUDIBLE_CUES
        and process.get("audible_end_cue_error") is None,
        "flow_preflight_contract": flow_preflight_contract,
        "host_time_provenance": host_time_provenance,
        "process_succeeded": process_outcome.get("succeeded") is True
        and process_outcome.get("timed_out") is False
        and _finite_number(process_outcome.get("elapsed_s"))
        and 0.0 <= float(process_outcome["elapsed_s"]) <= OVERALL_TIMEOUT_S
        and type(camera_outcome.get("returncode")) is int
        and camera_outcome["returncode"] == 0
        and type(telemetry_outcome.get("returncode")) is int
        and telemetry_outcome["returncode"] == 0,
        "non_actuating": process.get("controls_drone") is False,
        "log_files_present": logs_present,
        "packet_loss_free": logs_present and PACKET_LOSS_MESSAGE not in logs,
        "camera_complete": metadata.get("complete") is True
        and metadata.get("captured_frames") == REQUIRED_CAMERA_FRAMES
        and metadata.get("requested_frames") == REQUIRED_CAMERA_FRAMES
        and len(frames) == REQUIRED_CAMERA_FRAMES,
        "camera_contract": frames.dtype == np.uint8
        and frames.shape[1:] == (48, 64)
        and bool(np.all(frames % 17 == 0))
        and metadata.get("schema") == AIDECK_DECODED_CAPTURE_SCHEMA
        and metadata.get("transport") == "udp"
        and metadata.get("configured_source_endpoint")
        == REQUIRED_CAMERA_SOURCE_ENDPOINT
        and metadata.get("source_frame_contract") == REQUIRED_CAMERA_SOURCE_CONTRACT,
        "camera_rate": MINIMUM_CAMERA_RATE_HZ
        <= camera_rate
        <= MAXIMUM_CAMERA_RATE_HZ,
        "camera_cadence": bool(
            np.all(camera_host_gaps <= MAXIMUM_CAMERA_HOST_GAP_S)
        ),
        "camera_transport": drop_fraction <= 0.005
        and int(metadata["rejected_datagrams"]) == 0,
        "telemetry_rows": len(telemetry_times) >= 10,
        "telemetry_host_order": bool(np.all(telemetry_host_gaps > 0.0)),
        "telemetry_device_order": bool(np.all(telemetry_device_gaps > 0.0)),
        "telemetry_gap": bool(np.all(telemetry_host_gaps <= 0.075))
        and bool(np.all(telemetry_device_gaps <= 75.0)),
        "camera_telemetry_overlap": bool(
            telemetry_times[0] <= camera_times[0]
            and camera_times[-1] <= telemetry_times[-1]
        ),
        "camera_telemetry_proximity": float(nearest_gaps.max()) <= 0.030,
        "stationary_estimator": spans["stateEstimate.x"] <= 0.12
        and spans["stateEstimate.y"] <= 0.05
        and spans["stateEstimate.z"] <= 0.02
        and spans["stateEstimate.yaw"] <= 1.50,
        "battery": float(telemetry["pm.vbat"].min()) >= 3.70,
    }
    metrics = {
        "camera_frames": len(frames),
        "camera_rate_hz": camera_rate,
        "maximum_camera_host_gap_s": float(camera_host_gaps.max()),
        "p95_camera_host_gap_s": float(np.percentile(camera_host_gaps, 95)),
        "p99_camera_host_gap_s": float(np.percentile(camera_host_gaps, 99)),
        "camera_drop_fraction": drop_fraction,
        "telemetry_rows": len(telemetry_times),
        "maximum_telemetry_host_gap_s": float(telemetry_host_gaps.max()),
        "maximum_telemetry_device_gap_ms": float(telemetry_device_gaps.max()),
        "maximum_nearest_telemetry_gap_s": float(nearest_gaps.max()),
        "minimum_battery_v": float(telemetry["pm.vbat"].min()),
        "state_spans": spans,
    }
    return {
        "schema": SCHEMA,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "metrics": metrics,
        "paired_capture_passed": all(checks.values()),
        "capture_integrity_status": metadata.get("integrity_status", "unknown"),
        "synchronization_authority": False,
        "training_authority": False,
        "deployment_authority": False,
        "shadow_authority": False,
        "flight_authority": False,
        "authority_reason": (
            "This validates host-time proximity and a stationary transport envelope only; "
            "UDP source sequence/checksum and device-clock synchronization remain unproven."
        ),
    }


def _load_camera(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as artifact:
        frames = np.asarray(artifact["decoded_frames"])
        host_times = np.asarray(artifact["host_time_s"], dtype=np.float64)
        metadata = json.loads(str(artifact["metadata_json"]))
    if frames.ndim != 3 or len(frames) < 2 or host_times.shape != (len(frames),):
        raise ValueError("paired camera artifact has incompatible frame or timestamp shape")
    if not np.isfinite(host_times).all() or np.any(np.diff(host_times) <= 0.0):
        raise ValueError("paired camera host timestamps must be finite and strictly increasing")
    if not isinstance(metadata, dict):
        raise ValueError("paired camera metadata must be an object")
    return frames, host_times, metadata


def _load_telemetry(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = ("host_time_s", "crazyflie_time_ms", *REQUIRED_TELEMETRY)
    if tuple(reader.fieldnames or ()) != required:
        raise ValueError("paired telemetry must contain the exact columns in contract order")
    if len(rows) < 2:
        raise ValueError("paired telemetry is missing required rows")
    result = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in required
    }
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("paired telemetry contains nonfinite values")
    return result


def _nearest_gaps(camera_times: np.ndarray, telemetry_times: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(telemetry_times, camera_times)
    left = np.maximum(positions - 1, 0)
    right = np.minimum(positions, len(telemetry_times) - 1)
    return np.minimum(
        np.abs(camera_times - telemetry_times[left]),
        np.abs(telemetry_times[right] - camera_times),
    )


def _validate_embedded_flow_preflight(
    root: Path,
    process: dict[str, object],
) -> bool:
    evidence = process.get("flow_preflight_evidence")
    started = process.get("started_host_time_s")
    if not isinstance(evidence, dict) or not _finite_number(started):
        return False
    if (
        evidence.get("schema") != FLOW_PREFLIGHT_SCHEMA
        or evidence.get("embedded_name") != "flow_preflight_process.json"
        or not isinstance(evidence.get("source_path"), str)
        or not evidence["source_path"]
        or not _finite_number(evidence.get("age_s"))
    ):
        return False
    try:
        raw = (root / "flow_preflight_process.json").read_bytes()
        report = json.loads(raw)
        if not isinstance(report, dict):
            return False
        actual_age_s = validate_flow_preflight_report(report, now_s=float(started))
    except (OSError, json.JSONDecodeError, HardwareSafetyError):
        return False
    recorded_age_s = float(evidence["age_s"])
    return (
        evidence.get("sha256") == sha256(raw).hexdigest()
        and 0.0 <= recorded_age_s <= MAXIMUM_REPORT_AGE_S
        and recorded_age_s <= actual_age_s
        and actual_age_s - recorded_age_s <= 5.0
    )


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(float(value))
    )


def _validate_host_time_provenance(
    process: dict[str, object],
    camera_times: np.ndarray,
    telemetry_times: np.ndarray,
) -> bool:
    started = process.get("started_host_time_s")
    ended = process.get("ended_host_time_s")
    if not _finite_number(started) or not _finite_number(ended):
        return False
    started_s = float(started)
    ended_s = float(ended)
    return (
        0.0 <= ended_s - started_s <= OVERALL_TIMEOUT_S
        and started_s <= float(camera_times[0])
        and float(camera_times[-1]) <= ended_s
        and started_s <= float(telemetry_times[0])
        and float(telemetry_times[-1]) <= ended_s
    )
