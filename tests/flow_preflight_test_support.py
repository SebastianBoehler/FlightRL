from __future__ import annotations

import json
from pathlib import Path
from time import time


def passing_flow_preflight_report(
    *,
    started_s: float,
    ended_s: float,
    telemetry_uri: str = "usb://0",
) -> dict[str, object]:
    return {
        "schema": "flightrl.aideck_flow_preflight_process.v1",
        "controls_drone": False,
        "non_actuating": True,
        "props_off_required": True,
        "rigid_support_required": True,
        "flight_authority": False,
        "telemetry_uri": telemetry_uri,
        "deck_expectations": {
            "expect_ai_deck": True,
            "expect_flow_deck": True,
            "expect_multiranger": False,
            "expect_zranger": True,
        },
        "telemetry_variables": [
            "motion.motion",
            "motion.deltaX",
            "motion.deltaY",
            "motion.squal",
            "range.zrange",
        ],
        "telemetry_period_ms": 50,
        "telemetry_log_blocks": 1,
        "deck_check_timeout_s": 15.0,
        "telemetry_duration_s": 6.0,
        "telemetry_timeout_s": 15.0,
        "cleanup_timeout_s": 3.0,
        "audible_cues": {
            "motion_start": "/System/Library/Sounds/Glass.aiff",
            "success": "/System/Library/Sounds/Hero.aiff",
            "failure": "/System/Library/Sounds/Basso.aiff",
        },
        "started_host_time_s": started_s,
        "ended_host_time_s": ended_s,
        "process_outcome": {
            "succeeded": True,
            "deck_check": {"returncode": 0, "timed_out": False},
            "telemetry": {"returncode": 0, "timed_out": False},
            "validation_error": None,
            "packet_loss_free": True,
            "flow_preflight_passed": True,
        },
    }


def write_fresh_passing_flow_preflight(tmp_path: Path) -> Path:
    ended_s = time()
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(
            passing_flow_preflight_report(
                started_s=ended_s - 1.0,
                ended_s=ended_s,
            )
        )
    )
    return path
