from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .validation import require_bool, require_finite_real


@dataclass(frozen=True, slots=True)
class SixDofSensorProfile:
    name: str = "ideal"
    range_observation_enabled: bool = True
    state_noise_std_m: float = 0.0
    velocity_noise_std_m_s: float = 0.0
    body_rate_noise_std_rad_s: float = 0.0
    range_noise_std_m: float = 0.0
    range_dropout_prob: float = 0.0
    action_lag_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("sensor profile name must be a non-empty string")
        object.__setattr__(
            self,
            "range_observation_enabled",
            require_bool(
                self.range_observation_enabled,
                "range_observation_enabled",
            ),
        )
        for name in (
            "state_noise_std_m",
            "velocity_noise_std_m_s",
            "body_rate_noise_std_rad_s",
            "range_noise_std_m",
            "range_dropout_prob",
            "action_lag_s",
        ):
            object.__setattr__(
                self,
                name,
                require_finite_real(getattr(self, name), name, minimum=0.0),
            )
        if self.range_dropout_prob > 1.0:
            raise ValueError("range_dropout_prob must be at most 1")

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
        if not self.range_observation_enabled:
            return True
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
        dt = require_finite_real(
            dt,
            "sensor action dt",
            minimum=0.0,
            strictly_greater=True,
        )
        if self.action_lag_s <= 0.0:
            return 1.0
        return float(dt / (self.action_lag_s + dt))

    def as_env_values(self) -> dict[str, float | int]:
        return {
            "range_observation_enabled": int(self.range_observation_enabled),
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
NO_RANGER_SENSOR_PROFILE = SixDofSensorProfile(
    name="no_ranger",
    range_observation_enabled=False,
)
RANGER_SENSOR_PROFILE = SixDofSensorProfile(name="ranger")

BUILTIN_SENSOR_PROFILES = {
    "ideal": IDEAL_SENSOR_PROFILE,
    "none": IDEAL_SENSOR_PROFILE,
    "off": IDEAL_SENSOR_PROFILE,
    "ranger": RANGER_SENSOR_PROFILE,
    "no_ranger": NO_RANGER_SENSOR_PROFILE,
}


def resolve_sensor_profile(value: str | Path | SixDofSensorProfile | None) -> SixDofSensorProfile:
    if value is None:
        return IDEAL_SENSOR_PROFILE
    if isinstance(value, SixDofSensorProfile):
        return value
    if str(value) in BUILTIN_SENSOR_PROFILES:
        return BUILTIN_SENSOR_PROFILES[str(value)]
    path = Path(value)
    if not path.exists():
        expected = ", ".join(sorted(BUILTIN_SENSOR_PROFILES))
        raise ValueError(f"unknown 6-DoF sensor profile {value!r}; expected one of {expected} or a JSON path")
    data = json.loads(path.read_text())
    payload = data.get("sensor_profile", data)
    return SixDofSensorProfile(
        name=payload.get("name", path.stem),
        range_observation_enabled=payload.get("range_observation_enabled", True),
        state_noise_std_m=payload.get("state_noise_std_m", 0.0),
        velocity_noise_std_m_s=payload.get("velocity_noise_std_m_s", 0.0),
        body_rate_noise_std_rad_s=payload.get("body_rate_noise_std_rad_s", 0.0),
        range_noise_std_m=payload.get("range_noise_std_m", 0.0),
        range_dropout_prob=payload.get("range_dropout_prob", 0.0),
        action_lag_s=payload.get("action_lag_s", 0.0),
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
    if not profile.range_observation_enabled:
        return np.full_like(ranges_m, max_range_m, dtype=np.float32)
    observed = noisy_values(ranges_m, profile.range_noise_std_m, rng)
    if profile.range_dropout_prob > 0.0:
        dropout = rng.random(observed.shape) < profile.range_dropout_prob
        observed = np.where(dropout, max_range_m, observed)
    return np.clip(observed, 0.0, max_range_m).astype(np.float32)
