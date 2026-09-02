from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .physics import SixDofPhysicsProfile


EMBODIMENT_CONTRACT_ID = "flightrl.sixdof.embodiment.v1"
EMBODIMENT_FIELDS = (
    "mass_kg",
    "linear_drag",
    "rate_tau_s",
    "thrust_scale",
    "max_rate_roll_rad_s",
    "max_rate_pitch_rad_s",
    "max_rate_yaw_rad_s",
    "motor_tau_s",
)
EMBODIMENT_PHYSICS_COLUMNS = (0, 2, 3, 4, 5, 6, 7, 8)


@dataclass(frozen=True, slots=True)
class EmbodimentDescriptor:
    mass_kg: float
    linear_drag: float
    rate_tau_s: float
    thrust_scale: float
    max_rate_roll_rad_s: float
    max_rate_pitch_rad_s: float
    max_rate_yaw_rad_s: float
    motor_tau_s: float

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if any(not isfinite(value) for value in values):
            raise ValueError("embodiment values must be finite")
        if self.mass_kg <= 0.0 or self.thrust_scale <= 0.0:
            raise ValueError("embodiment mass and thrust scale must be positive")
        if min(*values[1:3], self.motor_tau_s) < 0.0:
            raise ValueError("embodiment drag and time constants cannot be negative")
        if min(values[4:7]) <= 0.0:
            raise ValueError("embodiment rate limits must be positive")

    @classmethod
    def from_physics_profile(
        cls,
        profile: SixDofPhysicsProfile,
    ) -> EmbodimentDescriptor:
        roll, pitch, yaw = profile.max_rate_rad_s
        return cls(
            mass_kg=profile.mass_kg,
            linear_drag=profile.linear_drag,
            rate_tau_s=profile.rate_tau_s,
            thrust_scale=profile.thrust_scale,
            max_rate_roll_rad_s=roll,
            max_rate_pitch_rad_s=pitch,
            max_rate_yaw_rad_s=yaw,
            motor_tau_s=profile.motor_tau_s,
        )

    def as_tuple(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in EMBODIMENT_FIELDS)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.as_tuple(), dtype=np.float32)


def embodiment_batch(physics_parameters: np.ndarray) -> np.ndarray:
    if (
        not isinstance(physics_parameters, np.ndarray)
        or physics_parameters.dtype != np.float32
        or physics_parameters.ndim != 2
        or physics_parameters.shape[1] != 9
        or not np.isfinite(physics_parameters).all()
    ):
        raise ValueError("physics parameters must be a finite float32 (N, 9) array")
    return np.ascontiguousarray(
        physics_parameters[:, EMBODIMENT_PHYSICS_COLUMNS],
        dtype=np.float32,
    )
