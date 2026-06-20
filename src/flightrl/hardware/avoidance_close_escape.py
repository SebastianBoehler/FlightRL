from __future__ import annotations

import math
from dataclasses import dataclass

from flightrl.hardware.avoidance_policy import AvoidanceCommand, RangerReading


@dataclass(frozen=True, slots=True)
class CloseEscapeCorrection:
    closest_side: str
    closest_range_m: float
    body_vx_m_s: float
    body_vy_m_s: float
    override_active: bool
    brake_active: bool

    def row(self) -> dict[str, float | str]:
        return {
            "close_escape_side": self.closest_side,
            "close_escape_range_m": self.closest_range_m,
            "body_vx_m_s": self.body_vx_m_s,
            "body_vy_m_s": self.body_vy_m_s,
            "closest_override_active": float(self.override_active),
            "velocity_brake_active": float(self.brake_active),
        }


def apply_close_escape_correction(
    command: AvoidanceCommand,
    reading: RangerReading,
    telemetry: dict[str, float],
    *,
    clearance_m: float,
    min_speed_m_s: float,
    brake_gain: float,
    brake_max_m_s: float,
) -> tuple[AvoidanceCommand, CloseEscapeCorrection]:
    side, distance = closest_horizontal_side(reading)
    body_vx, body_vy = body_velocity_m_s(telemetry)
    if clearance_m <= 0.0 or distance > clearance_m:
        return command, CloseEscapeCorrection(side, distance, body_vx, body_vy, False, False)

    vx, vy = command.vx_m_s, command.vy_m_s
    vx, vy, override = enforce_escape_speed(vx, vy, side, min_speed_m_s)
    vx, vy, brake = brake_toward_obstacle(
        vx,
        vy,
        side,
        body_vx,
        body_vy,
        gain=max(0.0, brake_gain),
        max_brake_m_s=max(0.0, brake_max_m_s),
    )
    corrected = AvoidanceCommand(vx, vy, command.yawrate_deg_s, command.zdistance_m)
    return corrected, CloseEscapeCorrection(side, distance, body_vx, body_vy, override, brake)


def closest_horizontal_side(reading: RangerReading) -> tuple[str, float]:
    values = {
        "front": reading.front_m,
        "back": reading.back_m,
        "left": reading.left_m,
        "right": reading.right_m,
    }
    side = min(values, key=values.__getitem__)
    return side, float(values[side])


def enforce_escape_speed(vx: float, vy: float, side: str, min_speed_m_s: float) -> tuple[float, float, bool]:
    speed = max(0.0, min_speed_m_s)
    if speed <= 0.0:
        return vx, vy, False
    if side == "front" and vx > -speed:
        return -speed, vy, True
    if side == "back" and vx < speed:
        return speed, vy, True
    if side == "left" and vy > -speed:
        return vx, -speed, True
    if side == "right" and vy < speed:
        return vx, speed, True
    return vx, vy, False


def brake_toward_obstacle(
    vx: float,
    vy: float,
    side: str,
    body_vx: float,
    body_vy: float,
    *,
    gain: float,
    max_brake_m_s: float,
) -> tuple[float, float, bool]:
    if gain <= 0.0 or max_brake_m_s <= 0.0:
        return vx, vy, False
    if side == "front" and body_vx > 0.0:
        return vx - min(max_brake_m_s, gain * body_vx), vy, True
    if side == "back" and body_vx < 0.0:
        return vx + min(max_brake_m_s, gain * abs(body_vx)), vy, True
    if side == "left" and body_vy > 0.0:
        return vx, vy - min(max_brake_m_s, gain * body_vy), True
    if side == "right" and body_vy < 0.0:
        return vx, vy + min(max_brake_m_s, gain * abs(body_vy)), True
    return vx, vy, False


def body_velocity_m_s(telemetry: dict[str, float]) -> tuple[float, float]:
    vx = _value(telemetry, "stateEstimate.vx")
    vy = _value(telemetry, "stateEstimate.vy")
    yaw_rad = math.radians(_value(telemetry, "stabilizer.yaw", fallback="stateEstimate.yaw"))
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    return cy * vx + sy * vy, -sy * vx + cy * vy


def _value(values: dict[str, float], key: str, *, fallback: str | None = None) -> float:
    raw = values.get(key)
    if raw is None and fallback is not None:
        raw = values.get(fallback)
    try:
        return float(raw if raw is not None else 0.0)
    except (TypeError, ValueError):
        return 0.0
