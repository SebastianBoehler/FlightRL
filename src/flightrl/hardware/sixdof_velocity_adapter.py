from __future__ import annotations

from dataclasses import dataclass
from math import degrees

import numpy as np

from flightrl.hardware.avoidance_policy import AvoidanceCommand, clip_horizontal_norm
from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.env import quat_to_yaw
from flightrl.sixdof.policies import roll_pitch_from_quat


@dataclass(frozen=True, slots=True)
class SixDofVelocityAdapterConfig:
    max_horizontal_speed_m_s: float = 0.18
    max_vertical_speed_m_s: float = 0.06
    max_yawrate_deg_s: float = 20.0
    rate_horizon_s: float = 0.08
    max_virtual_tilt_rad: float = 0.22
    horizontal_gain_s: float = 0.08
    policy_blend: float = 1.0


@dataclass(frozen=True, slots=True)
class SixDofVelocityCommand:
    vx_m_s: float
    vy_m_s: float
    vz_m_s: float
    yawrate_deg_s: float
    zdistance_m: float

    def as_avoidance_command(self) -> AvoidanceCommand:
        return AvoidanceCommand(self.vx_m_s, self.vy_m_s, self.yawrate_deg_s, self.zdistance_m)


def sixdof_action_to_velocity_command(
    env: SixDofCrazyflieEnv,
    action: np.ndarray,
    config: SixDofVelocityAdapterConfig,
    *,
    base: AvoidanceCommand | None = None,
) -> SixDofVelocityCommand:
    """Map one normalized six-DoF action to a guarded high-level hover command."""
    raw = np.asarray(action, dtype=np.float32).reshape(-1)
    if raw.shape[0] != 4:
        raise ValueError(f"expected 4D six-DoF action, got {raw.shape}")
    clipped = np.clip(raw, -1.0, 1.0)
    roll, pitch = roll_pitch_from_quat(env.quaternion[:1])
    max_rates = env.max_rate
    target_roll = float(np.clip(roll[0] + config.rate_horizon_s * clipped[1] * max_rates[0], -config.max_virtual_tilt_rad, config.max_virtual_tilt_rad))
    target_pitch = float(np.clip(pitch[0] + config.rate_horizon_s * clipped[2] * max_rates[1], -config.max_virtual_tilt_rad, config.max_virtual_tilt_rad))
    ax_body = env.gravity * target_pitch
    ay_body = -env.gravity * target_roll
    yaw = float(quat_to_yaw(env.quaternion[:1])[0])
    vx = config.horizontal_gain_s * (np.cos(yaw) * ax_body - np.sin(yaw) * ay_body)
    vy = config.horizontal_gain_s * (np.sin(yaw) * ax_body + np.cos(yaw) * ay_body)
    vz = float(np.clip(clipped[0] * config.max_vertical_speed_m_s, -config.max_vertical_speed_m_s, config.max_vertical_speed_m_s))
    yawrate = float(np.clip(degrees(clipped[3] * max_rates[2]), -config.max_yawrate_deg_s, config.max_yawrate_deg_s))
    policy = clip_horizontal_norm(
        AvoidanceCommand(float(vx), float(vy), yawrate, env.target_position[0, 2]),
        max_speed=config.max_horizontal_speed_m_s,
        max_yawrate=config.max_yawrate_deg_s,
    )
    if base is None or config.policy_blend >= 1.0:
        return SixDofVelocityCommand(policy.vx_m_s, policy.vy_m_s, vz, policy.yawrate_deg_s, policy.zdistance_m)
    blend = float(np.clip(config.policy_blend, 0.0, 1.0))
    mixed = AvoidanceCommand(
        (1.0 - blend) * base.vx_m_s + blend * policy.vx_m_s,
        (1.0 - blend) * base.vy_m_s + blend * policy.vy_m_s,
        (1.0 - blend) * base.yawrate_deg_s + blend * policy.yawrate_deg_s,
        base.zdistance_m,
    )
    guarded = clip_horizontal_norm(mixed, max_speed=config.max_horizontal_speed_m_s, max_yawrate=config.max_yawrate_deg_s)
    return SixDofVelocityCommand(guarded.vx_m_s, guarded.vy_m_s, blend * vz, guarded.yawrate_deg_s, guarded.zdistance_m)
