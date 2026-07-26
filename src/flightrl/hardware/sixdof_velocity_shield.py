from __future__ import annotations

from dataclasses import dataclass

from flightrl.hardware.avoidance_live import range_rate_row, update_range_rate
from flightrl.hardware.avoidance_policy import RangerReading, min_horizontal_range_m, reactive_clearance_command, reading_from_telemetry
from flightrl.hardware.avoidance_ttc import min_horizontal_ttc_s
from flightrl.hardware.sixdof_velocity_adapter import SixDofVelocityCommand


@dataclass(frozen=True, slots=True)
class SixDofVelocityShieldConfig:
    clearance_m: float = 0.45
    hard_clearance_m: float = 0.10
    max_speed_m_s: float = 0.35
    ttc_horizon_s: float = 0.80
    ttc_hard_s: float = 0.15
    ttc_gain: float = 1.2
    range_rate_alpha: float = 0.65
    range_rate_max_m_s: float = 5.0


@dataclass(slots=True)
class SixDofVelocityShieldState:
    previous_reading: RangerReading | None = None
    previous_time_s: float | None = None
    range_rate: RangerReading | None = None


@dataclass(frozen=True, slots=True)
class SixDofVelocityShieldResult:
    command: SixDofVelocityCommand
    active: bool
    min_horizontal_range_m: float
    min_horizontal_ttc_s: float
    range_rate: RangerReading | None


def apply_sixdof_velocity_shield(
    command: SixDofVelocityCommand,
    telemetry: dict,
    *,
    now_s: float,
    target_height_m: float,
    config: SixDofVelocityShieldConfig,
    state: SixDofVelocityShieldState,
) -> SixDofVelocityShieldResult:
    reading = reading_from_telemetry(telemetry)
    state.range_rate = update_range_rate(
        reading,
        state.previous_reading,
        state.previous_time_s,
        now_s,
        state.range_rate,
        alpha=config.range_rate_alpha,
        max_abs_rate_m_s=config.range_rate_max_m_s,
    )
    state.previous_reading = reading
    state.previous_time_s = now_s
    horizontal_range = min_horizontal_range_m(reading)
    horizontal_ttc = min_horizontal_ttc_s(reading, state.range_rate)
    ttc_active = config.ttc_horizon_s > config.ttc_hard_s and horizontal_ttc < config.ttc_horizon_s
    active = config.clearance_m > 0.0 and (horizontal_range < config.clearance_m or ttc_active)
    if not active:
        return SixDofVelocityShieldResult(command, False, horizontal_range, horizontal_ttc, state.range_rate)
    escape = reactive_clearance_command(
        reading,
        range_rate_m_s=state.range_rate,
        clearance_m=config.clearance_m,
        hard_clearance_m=config.hard_clearance_m,
        target_height_m=target_height_m,
        max_speed_m_s=config.max_speed_m_s,
        ttc_horizon_s=config.ttc_horizon_s,
        ttc_hard_s=config.ttc_hard_s,
        ttc_gain=config.ttc_gain,
    )
    return SixDofVelocityShieldResult(
        SixDofVelocityCommand(escape.vx_m_s, escape.vy_m_s, command.vz_m_s, command.yawrate_deg_s, command.zdistance_m),
        True,
        horizontal_range,
        horizontal_ttc,
        state.range_rate,
    )


def sixdof_shield_row(result: SixDofVelocityShieldResult) -> dict[str, float | bool]:
    return {
        "shield_active": result.active,
        "min_horizontal_range_m": result.min_horizontal_range_m,
        "min_horizontal_ttc_s": result.min_horizontal_ttc_s,
        **range_rate_row(result.range_rate),
    }
