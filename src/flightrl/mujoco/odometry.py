from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flightrl.sixdof.env import quat_to_yaw, wrap_angle


@dataclass(frozen=True, slots=True)
class OdometryNoiseConfig:
    position_noise_std_m: float = 0.01
    position_drift_std_m_per_step: float = 0.0008
    yaw_noise_std_rad: float = 0.01
    yaw_drift_std_rad_per_step: float = 0.0008


class SimulatedOdometry:
    def __init__(
        self,
        num_agents: int,
        config: OdometryNoiseConfig | None = None,
    ) -> None:
        self.config = config or OdometryNoiseConfig()
        self.position_drift = np.zeros((num_agents, 2), dtype=np.float32)
        self.yaw_drift = np.zeros(num_agents, dtype=np.float32)
        self.position_xy = np.zeros((num_agents, 2), dtype=np.float32)
        self.yaw = np.zeros(num_agents, dtype=np.float32)

    def reset(
        self,
        sim,
        rng: np.random.Generator,
        mask: np.ndarray,
    ) -> None:
        count = int(np.sum(mask))
        self.position_drift[mask] = rng.normal(
            0.0,
            self.config.position_noise_std_m,
            (count, 2),
        )
        self.yaw_drift[mask] = rng.normal(
            0.0,
            self.config.yaw_noise_std_rad,
            count,
        )
        self._write_estimate(sim, rng, mask)

    def advance(self, sim, rng: np.random.Generator) -> None:
        self.position_drift += rng.normal(
            0.0,
            self.config.position_drift_std_m_per_step,
            self.position_drift.shape,
        )
        self.yaw_drift += rng.normal(
            0.0,
            self.config.yaw_drift_std_rad_per_step,
            self.yaw_drift.shape,
        )
        self._write_estimate(sim, rng, np.ones(len(self.yaw), dtype=bool))

    def _write_estimate(
        self,
        sim,
        rng: np.random.Generator,
        mask: np.ndarray,
    ) -> None:
        count = int(np.sum(mask))
        self.position_xy[mask] = (
            sim.position[mask, :2]
            + self.position_drift[mask]
            + rng.normal(0.0, self.config.position_noise_std_m, (count, 2))
        )
        yaw = quat_to_yaw(sim.quaternion[mask])
        self.yaw[mask] = wrap_angle(
            yaw
            + self.yaw_drift[mask]
            + rng.normal(0.0, self.config.yaw_noise_std_rad, count)
        )
