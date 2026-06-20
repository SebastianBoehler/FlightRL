from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class SixDofSensorProfile:
    name: str = "ideal"
    state_noise_std_m: float = 0.0
    velocity_noise_std_m_s: float = 0.0
    body_rate_noise_std_rad_s: float = 0.0
    range_noise_std_m: float = 0.0
    range_dropout_prob: float = 0.0
    action_lag_s: float = 0.0

    @property
    def enabled(self) -> bool:
        return any(
            value > 0.0
            for value in (
                self.state_noise_std_m,
                self.velocity_noise_std_m_s,
                self.body_rate_noise_std_rad_s,
                self.range_noise_std_m,
                self.range_dropout_prob,
                self.action_lag_s,
            )
        )

    @property
    def observation_enabled(self) -> bool:
        return any(
            value > 0.0
            for value in (
                self.state_noise_std_m,
                self.velocity_noise_std_m_s,
                self.body_rate_noise_std_rad_s,
                self.range_noise_std_m,
                self.range_dropout_prob,
            )
        )

    def action_alpha(self, dt: float) -> float:
        if self.action_lag_s <= 0.0:
            return 1.0
        return float(dt / (self.action_lag_s + dt))

    def as_env_values(self) -> dict[str, float]:
        return {
            "state_noise_std_m": self.state_noise_std_m,
            "velocity_noise_std_m_s": self.velocity_noise_std_m_s,
            "body_rate_noise_std_rad_s": self.body_rate_noise_std_rad_s,
            "range_noise_std_m": self.range_noise_std_m,
            "range_dropout_prob": self.range_dropout_prob,
            "action_lag_s": self.action_lag_s,
        }

    def as_report(self) -> dict[str, float | str | bool]:
        return {**asdict(self), "enabled": self.enabled}


IDEAL_SENSOR_PROFILE = SixDofSensorProfile()


def resolve_sensor_profile(value: str | Path | SixDofSensorProfile | None) -> SixDofSensorProfile:
    if value is None or value in {"ideal", "none", "off"}:
        return IDEAL_SENSOR_PROFILE
    if isinstance(value, SixDofSensorProfile):
        return value
    path = Path(value)
    data = json.loads(path.read_text())
    payload = data.get("sensor_profile", data)
    return SixDofSensorProfile(
        name=str(payload.get("name", path.stem)),
        state_noise_std_m=float(payload.get("state_noise_std_m", 0.0) or 0.0),
        velocity_noise_std_m_s=float(payload.get("velocity_noise_std_m_s", 0.0) or 0.0),
        body_rate_noise_std_rad_s=float(payload.get("body_rate_noise_std_rad_s", 0.0) or 0.0),
        range_noise_std_m=float(payload.get("range_noise_std_m", 0.0) or 0.0),
        range_dropout_prob=float(payload.get("range_dropout_prob", 0.0) or 0.0),
        action_lag_s=float(payload.get("action_lag_s", 0.0) or 0.0),
    )


def noisy_values(values: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0.0:
        return values
    return values + rng.normal(0.0, std, size=values.shape).astype(np.float32)


def observed_ranges(
    ranges_m: np.ndarray,
    *,
    max_range_m: float,
    profile: SixDofSensorProfile,
    rng: np.random.Generator,
) -> np.ndarray:
    observed = noisy_values(ranges_m, profile.range_noise_std_m, rng)
    if profile.range_dropout_prob > 0.0:
        dropout = rng.random(observed.shape) < profile.range_dropout_prob
        observed = np.where(dropout, max_range_m, observed)
    return np.clip(observed, 0.0, max_range_m).astype(np.float32)
