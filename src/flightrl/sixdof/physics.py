from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PHYSICS_DIM = 9
MASS = 0
GRAVITY = 1
LINEAR_DRAG = 2
RATE_TAU = 3
THRUST_SCALE = 4
MAX_RATE_ROLL = 5
MAX_RATE_PITCH = 6
MAX_RATE_YAW = 7
MOTOR_TAU = 8


@dataclass(frozen=True, slots=True)
class SixDofPhysicsProfile:
    mass_kg: float = 0.036
    gravity_m_s2: float = 9.81
    linear_drag: float = 0.10
    rate_tau_s: float = 0.045
    thrust_scale: float = 0.75
    max_rate_rad_s: tuple[float, float, float] = (6.0, 6.0, 4.0)
    motor_tau_s: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.mass_kg,
                self.gravity_m_s2,
                self.linear_drag,
                self.rate_tau_s,
                self.thrust_scale,
                *self.max_rate_rad_s,
                self.motor_tau_s,
            ],
            dtype=np.float32,
        )

    def as_report(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SixDofDomainRandomization:
    mass_scale: tuple[float, float] = (1.0, 1.0)
    linear_drag_scale: tuple[float, float] = (1.0, 1.0)
    rate_tau_scale: tuple[float, float] = (1.0, 1.0)
    thrust_scale_scale: tuple[float, float] = (1.0, 1.0)
    max_rate_scale: tuple[float, float] = (1.0, 1.0)
    motor_tau_s: tuple[float, float] | None = None

    @property
    def enabled(self) -> bool:
        return any(
            lo != hi
            for lo, hi in (
                self.mass_scale,
                self.linear_drag_scale,
                self.rate_tau_scale,
                self.thrust_scale_scale,
                self.max_rate_scale,
            )
        ) or self.motor_tau_s is not None


LEGACY_PHYSICS = SixDofPhysicsProfile()
CRAZYFLIE_BRUSHLESS_PHYSICS = SixDofPhysicsProfile(linear_drag=0.08, motor_tau_s=0.035)
PUFFER_DRONE_PHYSICS = SixDofPhysicsProfile(
    mass_kg=0.027,
    linear_drag=0.0,
    max_rate_rad_s=(20.0, 20.0, 20.0),
    motor_tau_s=0.150,
)
CRAZYFLIE_TRAINING_RANDOMIZATION = SixDofDomainRandomization(
    mass_scale=(0.92, 1.10),
    linear_drag_scale=(0.75, 1.75),
    rate_tau_scale=(0.75, 1.45),
    thrust_scale_scale=(0.88, 1.16),
    max_rate_scale=(0.85, 1.15),
    motor_tau_s=(0.015, 0.060),
)


def resolve_physics_profile(value: str | SixDofPhysicsProfile | None) -> SixDofPhysicsProfile:
    if value is None or value == "legacy":
        return LEGACY_PHYSICS
    if value == "crazyflie_brushless":
        return CRAZYFLIE_BRUSHLESS_PHYSICS
    if value == "puffer_drone":
        return PUFFER_DRONE_PHYSICS
    if isinstance(value, SixDofPhysicsProfile):
        return value
    path = Path(value)
    if path.exists():
        data = json.loads(path.read_text())
        payload = data.get("physics_profile", data)
        default = LEGACY_PHYSICS
        rates = payload.get("max_rate_rad_s", default.max_rate_rad_s)
        return SixDofPhysicsProfile(
            mass_kg=float(payload.get("mass_kg", default.mass_kg)),
            gravity_m_s2=float(payload.get("gravity_m_s2", default.gravity_m_s2)),
            linear_drag=float(payload.get("linear_drag", default.linear_drag)),
            rate_tau_s=float(payload.get("rate_tau_s", default.rate_tau_s)),
            thrust_scale=float(payload.get("thrust_scale", default.thrust_scale)),
            max_rate_rad_s=tuple(float(item) for item in rates),
            motor_tau_s=float(payload.get("motor_tau_s", default.motor_tau_s)),
        )
    raise ValueError(f"unknown 6-DoF physics profile {value!r}")


def resolve_domain_randomization(value: str | SixDofDomainRandomization | None) -> SixDofDomainRandomization:
    if value is None or value in {"none", "off", "disabled"}:
        return SixDofDomainRandomization()
    if value == "crazyflie_training":
        return CRAZYFLIE_TRAINING_RANDOMIZATION
    if isinstance(value, SixDofDomainRandomization):
        return value
    raise ValueError(f"unknown 6-DoF domain randomization profile {value!r}")


def sample_physics(
    profile: SixDofPhysicsProfile,
    randomization: SixDofDomainRandomization,
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    base = np.repeat(profile.as_array()[None, :], count, axis=0)
    if not randomization.enabled:
        return base
    base[:, MASS] *= rng.uniform(*randomization.mass_scale, size=count)
    base[:, LINEAR_DRAG] *= rng.uniform(*randomization.linear_drag_scale, size=count)
    base[:, RATE_TAU] *= rng.uniform(*randomization.rate_tau_scale, size=count)
    base[:, THRUST_SCALE] *= rng.uniform(*randomization.thrust_scale_scale, size=count)
    rate_scale = rng.uniform(*randomization.max_rate_scale, size=(count, 1))
    base[:, MAX_RATE_ROLL : MAX_RATE_YAW + 1] *= rate_scale
    if randomization.motor_tau_s is not None:
        base[:, MOTOR_TAU] = rng.uniform(*randomization.motor_tau_s, size=count)
    return base.astype(np.float32)
