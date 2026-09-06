"""Published size/mass references; controller dynamics are explicit surrogates."""

from dataclasses import dataclass
import numpy as np
from flightrl.sixdof.physics import SixDofPhysicsProfile


@dataclass(frozen=True)
class Vehicle:
    name: str
    mass_kg: float
    dimensions_m: tuple[float, float, float]
    source: str
    envelope_note: str
    motor_tau_s: float
    rate_tau_s: float

    @property
    def radius(self):
        # Orientation-independent enclosing sphere; includes vertical clearance.
        return float(np.linalg.norm(self.dimensions_m) / 2)

    def physics(self):
        return SixDofPhysicsProfile(
            mass_kg=self.mass_kg,
            motor_tau_s=self.motor_tau_s,
            rate_tau_s=self.rate_tau_s,
            max_rate_rad_s=(2, 2, 1.5),
        ).as_array()


VEHICLES = {
    "fpv": Vehicle(
        "Avata 2 size reference",
        0.377,
        (0.185, 0.212, 0.064),
        "https://www.dji.com/avata-2/specs",
        "Published dimensions",
        0.04,
        0.08,
    ),
    "industrial": Vehicle(
        "Matrice 350 RTK size reference",
        6.47,
        (1.344, 1.204, 0.43),
        "https://enterprise.dji.com/matrice-350-rtk/specs",
        "Conservative assumed propeller envelope: published 0.810 x 0.670 body plus 0.534 m allowance",
        0.10,
        0.16,
    ),
    "agriculture": Vehicle(
        "Agras T25 unloaded size reference",
        32,
        (2.585, 2.675, 0.780),
        "https://ag.dji.com/t25/specs",
        "Published unfolded dimensions, battery installed, no payload",
        0.16,
        0.25,
    ),
}
# Time constants are experiment assumptions, not identified DJI flight dynamics.
