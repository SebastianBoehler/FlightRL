from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from flow_preflight_test_support import passing_flow_preflight_report


def write_valid_paired_run(
    tmp_path: Path,
    *,
    telemetry_start_s: float = 100.0,
    camera_times: np.ndarray | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if camera_times is None:
        camera_times = 100.0 + np.arange(1200, dtype=np.float64) / 65.0
    frames = np.full((1200, 48, 64), 17, dtype=np.uint8)
    metadata = {
        "schema": "flightrl.aideck_decoded_frame_capture.v2",
        "complete": True,
        "captured_frames": 1200,
        "requested_frames": 1200,
        "dropped_frames": 0,
        "rejected_datagrams": 0,
        "integrity_status": "unreviewed",
        "transport": "udp",
        "configured_source_endpoint": {"host": "192.168.4.1", "port": 5000},
        "source_frame_contract": {
            "width": 64,
            "height": 48,
            "depth": 1,
            "format": 2,
            "encoding": "packed_gray4_even_high_odd_low",
        },
    }
    np.savez_compressed(
        run_dir / "decoded_frames.npz",
        decoded_frames=frames,
        host_time_s=camera_times,
        metadata_json=np.asarray(json.dumps(metadata)),
        complete=np.asarray(True),
        dropped_frames=np.asarray(0),
        rejected_datagrams=np.asarray(0),
    )
    rows = [
        "host_time_s,crazyflie_time_ms,stateEstimate.x,stateEstimate.y,"
        "stateEstimate.z,stateEstimate.yaw,pm.vbat"
    ]
    for index in range(461):
        rows.append(
            f"{telemetry_start_s + index * 0.05:.6f},{index * 50},"
            f"{index * 0.0001},{index * 0.00005},0.31,{index * 0.001},4.05"
        )
    (run_dir / "telemetry.csv").write_text("\n".join(rows) + "\n")
    preflight = json.dumps(
        passing_flow_preflight_report(started_s=88.0, ended_s=95.0),
        sort_keys=True,
    ).encode()
    (run_dir / "flow_preflight_process.json").write_bytes(preflight)
    (run_dir / "capture_process.json").write_text(
        json.dumps(_capture_manifest(run_dir, preflight))
    )
    (run_dir / "camera.log").write_text("")
    (run_dir / "telemetry.log").write_text("")
    return run_dir


def _capture_manifest(run_dir: Path, preflight: bytes) -> dict[str, object]:
    return {
        "schema": "flightrl.aideck_bounded_paired_capture.v1",
        "controls_drone": False,
        "non_actuating": True,
        "started_host_time_s": 99.0,
        "ended_host_time_s": 124.0,
        "telemetry_uri": "usb://0",
        "telemetry_period_ms": 50,
        "telemetry_duration_s": 23.0,
        "telemetry_variables": [
            "stateEstimate.x",
            "stateEstimate.y",
            "stateEstimate.z",
            "stateEstimate.yaw",
            "pm.vbat",
        ],
        "telemetry_log_blocks": 1,
        "camera_requested_frames": 1200,
        "camera_source_endpoint": {"host": "192.168.4.1", "port": 5000},
        "camera_bind_port": 5001,
        "telemetry_ready_path": str(run_dir / "telemetry.csv"),
        "telemetry_ready_columns": [
            "host_time_s",
            "crazyflie_time_ms",
            "stateEstimate.x",
            "stateEstimate.y",
            "stateEstimate.z",
            "stateEstimate.yaw",
            "pm.vbat",
        ],
        "telemetry_ready_minimums": {"pm.vbat": 3.70},
        "telemetry_ready_timeout_s": 10.0,
        "cleanup_timeout_s": 3.0,
        "overall_timeout_s": 38.0,
        "flow_preflight_required_for_live": True,
        "audible_cues": {
            "motion_start": "/System/Library/Sounds/Glass.aiff",
            "success": "/System/Library/Sounds/Hero.aiff",
            "failure": "/System/Library/Sounds/Basso.aiff",
        },
        "flow_preflight_evidence": {
            "schema": "flightrl.aideck_flow_preflight_process.v1",
            "source_path": "/operator/preflight_process.json",
            "embedded_name": "flow_preflight_process.json",
            "sha256": sha256(preflight).hexdigest(),
            "age_s": 4.0,
        },
        "process_outcome": {
            "succeeded": True,
            "timed_out": False,
            "elapsed_s": 25.0,
            "camera": {"returncode": 0},
            "telemetry": {"returncode": 0},
        },
    }
