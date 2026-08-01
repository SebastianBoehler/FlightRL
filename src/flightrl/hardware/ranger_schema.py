from __future__ import annotations


RANGER_KEYS = (
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "range.zrange",
)
HORIZONTAL_RANGER_KEYS = RANGER_KEYS[:4]
RANGER_POSE_KEYS = (
    "host_time_s",
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
)
