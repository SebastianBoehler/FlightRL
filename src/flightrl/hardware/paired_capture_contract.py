from __future__ import annotations

from .aideck_protocol import (
    AIDECK_DECODED_CAPTURE_SCHEMA as AIDECK_DECODED_CAPTURE_SCHEMA,
)
from .aideck_protocol import AIDECK_GRAY4_FORMAT


PROCESS_SCHEMA = "flightrl.aideck_bounded_paired_capture.v1"
REQUIRED_TELEMETRY_URI = "usb://0"
REQUIRED_CAMERA_SOURCE_ENDPOINT = {"host": "192.168.4.1", "port": 5000}
REQUIRED_CAMERA_BIND_PORT = 5001
REQUIRED_CAMERA_FRAMES = 1200
TELEMETRY_DURATION_S = 23.0
TELEMETRY_PERIOD_MS = 50
MINIMUM_CAMERA_RATE_HZ = 55.0
MAXIMUM_CAMERA_RATE_HZ = 75.0
MAXIMUM_CAMERA_HOST_GAP_S = 0.075
MINIMUM_BATTERY_V = 3.70
TELEMETRY_TAIL_S = 1.0
TELEMETRY_READY_TIMEOUT_S = 10.0
CLEANUP_TIMEOUT_S = 3.0
OVERALL_TIMEOUT_S = 38.0
REQUIRED_TELEMETRY = (
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.yaw",
    "pm.vbat",
)
REQUIRED_TELEMETRY_COLUMNS = (
    "host_time_s",
    "crazyflie_time_ms",
    *REQUIRED_TELEMETRY,
)
REQUIRED_CAMERA_SOURCE_CONTRACT = {
    "width": 64,
    "height": 48,
    "depth": 1,
    "format": AIDECK_GRAY4_FORMAT,
    "encoding": "packed_gray4_even_high_odd_low",
}
REQUIRED_DECK_EXPECTATIONS = {
    "expect_flow_deck": True,
    "expect_multiranger": False,
    "expect_ai_deck": True,
    "expect_zranger": True,
}
