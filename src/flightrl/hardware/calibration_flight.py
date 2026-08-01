from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


CALIBRATION_PATTERNS = ("line", "yaw", "square", "line_yaw_square", "full_yaw_square")


CALIBRATION_LOG_VARIABLES = (
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "range.zrange",
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.vz",
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "gyro.x",
    "gyro.y",
    "gyro.z",
    "pm.vbat",
)


@dataclass(frozen=True, slots=True)
class CalibrationCommand:
    mode: str
    duration_s: float
    vx_m_s: float = 0.0
    vy_m_s: float = 0.0
    vz_m_s: float = 0.0
    yawrate_deg_s: float = 0.0


def build_calibration_sequence(
    *,
    pattern: str = "line_yaw_square",
    segment_s: float = 1.6,
    hover_s: float = 1.0,
    speed_m_s: float = 0.12,
    yawrate_deg_s: float = 20.0,
) -> list[CalibrationCommand]:
    _bounded("segment_s", segment_s, 0.1, 10.0)
    _bounded("hover_s", hover_s, 0.0, 10.0)
    _bounded("speed_m_s", speed_m_s, 0.05, 0.5)
    _bounded("yawrate_deg_s", yawrate_deg_s, 5.0, 120.0)
    sequence = [CalibrationCommand("hover_start", hover_s)]
    if pattern == "full_yaw_square":
        sequence.append(
            CalibrationCommand(
                "yaw_360",
                duration_s=360.0 / yawrate_deg_s,
                yawrate_deg_s=yawrate_deg_s,
            )
        )
        sequence.extend(square_commands(segment_s, speed_m_s))
        sequence.append(CalibrationCommand("hover_end", hover_s))
        return sequence
    if pattern in {"line", "line_yaw_square"}:
        sequence.extend(line_commands(segment_s, speed_m_s))
    if pattern in {"yaw", "line_yaw_square"}:
        sequence.extend(yaw_commands(segment_s, yawrate_deg_s))
    if pattern in {"square", "line_yaw_square"}:
        sequence.extend(square_commands(segment_s, speed_m_s))
    if len(sequence) == 1:
        raise ValueError(f"unknown calibration pattern {pattern!r}")
    sequence.append(CalibrationCommand("hover_end", hover_s))
    return sequence


def line_commands(segment_s: float, speed_m_s: float) -> list[CalibrationCommand]:
    return [
        CalibrationCommand("line_x_pos", segment_s, vx_m_s=speed_m_s),
        CalibrationCommand("line_x_neg", segment_s, vx_m_s=-speed_m_s),
        CalibrationCommand("line_y_pos", segment_s, vy_m_s=speed_m_s),
        CalibrationCommand("line_y_neg", segment_s, vy_m_s=-speed_m_s),
    ]


def yaw_commands(segment_s: float, yawrate_deg_s: float) -> list[CalibrationCommand]:
    return [
        CalibrationCommand("yaw_pos", segment_s, yawrate_deg_s=yawrate_deg_s),
        CalibrationCommand("yaw_neg", segment_s, yawrate_deg_s=-yawrate_deg_s),
    ]


def square_commands(segment_s: float, speed_m_s: float) -> list[CalibrationCommand]:
    return [
        CalibrationCommand("square_x_pos", segment_s, vx_m_s=speed_m_s),
        CalibrationCommand("square_y_pos", segment_s, vy_m_s=speed_m_s),
        CalibrationCommand("square_x_neg", segment_s, vx_m_s=-speed_m_s),
        CalibrationCommand("square_y_neg", segment_s, vy_m_s=-speed_m_s),
    ]


def sequence_duration_s(sequence: list[CalibrationCommand]) -> float:
    return float(sum(command.duration_s for command in sequence))


def command_row(command: CalibrationCommand) -> dict[str, float | str]:
    return {
        "mode": command.mode,
        "vx_m_s": command.vx_m_s,
        "vy_m_s": command.vy_m_s,
        "vz_m_s": command.vz_m_s,
        "yawrate_deg_s": command.yawrate_deg_s,
    }


def _bounded(name: str, value: float, low: float, high: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    if not isfinite(float(value)) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
