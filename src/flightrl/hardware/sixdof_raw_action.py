from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RawPufferActionConfig:
    hover_thrust_percent: float = 49.0
    thrust_scale: float = 0.75
    max_roll_rate_deg_s: float = 343.7747
    max_pitch_rate_deg_s: float = 343.7747
    max_yaw_rate_deg_s: float = 229.1831


@dataclass(frozen=True, slots=True)
class RawManualSetpoint:
    roll_rate_deg_s: float
    pitch_rate_deg_s: float
    commander_pitch_rate_deg_s: float
    yaw_rate_deg_s: float
    thrust_percent: float


def raw_action_to_manual_setpoint(action: np.ndarray, config: RawPufferActionConfig) -> RawManualSetpoint:
    raw = np.asarray(action, dtype=np.float32).reshape(-1)
    if raw.shape[0] != 4:
        raise ValueError(f"expected 4D Puffer action, got {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise ValueError("Puffer action contains non-finite values")

    thrust_percent = float(config.hover_thrust_percent * (1.0 + config.thrust_scale * float(raw[0])))
    if not 0.0 <= thrust_percent <= 100.0:
        raise ValueError(f"mapped thrust_percent is outside Crazyflie range: {thrust_percent:.3f}")
    pitch_rate = float(raw[2] * config.max_pitch_rate_deg_s)
    return RawManualSetpoint(
        roll_rate_deg_s=float(raw[1] * config.max_roll_rate_deg_s),
        pitch_rate_deg_s=pitch_rate,
        commander_pitch_rate_deg_s=-pitch_rate,
        yaw_rate_deg_s=float(raw[3] * config.max_yaw_rate_deg_s),
        thrust_percent=thrust_percent,
    )
