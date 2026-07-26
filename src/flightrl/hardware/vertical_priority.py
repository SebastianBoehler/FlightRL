from __future__ import annotations

from flightrl.hardware.avoidance_policy import AvoidanceCommand, RangerReading, min_horizontal_range_m, min_horizontal_ttc_s


def apply_vertical_priority(
    command: AvoidanceCommand,
    reading: RangerReading,
    range_rate: RangerReading | None,
    *,
    top_clearance_m: float,
    bottom_clearance_m: float,
    horizontal_escape_clearance_m: float,
    horizontal_hard_ttc_s: float,
) -> tuple[AvoidanceCommand, bool]:
    vertical_pressure = reading.up_m < top_clearance_m or reading.zrange_m < bottom_clearance_m
    if not vertical_pressure:
        return command, False
    if min_horizontal_range_m(reading) <= horizontal_escape_clearance_m:
        return command, False
    if min_horizontal_ttc_s(reading, range_rate) <= horizontal_hard_ttc_s:
        return command, False
    return AvoidanceCommand(0.0, 0.0, command.yawrate_deg_s, command.zdistance_m), True
