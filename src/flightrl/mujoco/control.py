from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from flightrl.sixdof.disturbance import disturbance_accel
from flightrl.sixdof.physics import SixDofPhysicsProfile
from flightrl.sixdof.validation import require_finite_real, require_real_tuple


@dataclass(frozen=True, slots=True)
class MuJoCoControlParams:
    mass_kg: float = 0.036
    gravity: float = 9.81
    linear_drag: float = 0.10
    max_rate_rad_s: tuple[float, float, float] = (6.0, 6.0, 4.0)
    rate_kp: float = 2.4e-4
    rate_kd: float = 3.0e-5
    thrust_scale: float = 0.75
    rate_tau_s: float = 0.045
    motor_tau_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("mass_kg", "gravity", "thrust_scale"):
            object.__setattr__(
                self,
                name,
                require_finite_real(
                    getattr(self, name),
                    name,
                    minimum=0.0,
                    strictly_greater=True,
                ),
            )
        for name in (
            "linear_drag",
            "rate_kp",
            "rate_kd",
            "rate_tau_s",
            "motor_tau_s",
        ):
            object.__setattr__(
                self,
                name,
                require_finite_real(getattr(self, name), name, minimum=0.0),
            )
        object.__setattr__(
            self,
            "max_rate_rad_s",
            require_real_tuple(
                self.max_rate_rad_s,
                "max_rate_rad_s",
                3,
                minimum=0.0,
                strictly_greater=True,
            ),
        )


def control_from_physics_profile(
    profile: SixDofPhysicsProfile,
) -> MuJoCoControlParams:
    return MuJoCoControlParams(
        mass_kg=profile.mass_kg,
        gravity=profile.gravity_m_s2,
        linear_drag=profile.linear_drag,
        max_rate_rad_s=profile.max_rate_rad_s,
        thrust_scale=profile.thrust_scale,
        rate_tau_s=profile.rate_tau_s,
        motor_tau_s=profile.motor_tau_s,
    )


def resolve_control(
    control: MuJoCoControlParams | None,
    profile: SixDofPhysicsProfile,
) -> MuJoCoControlParams:
    expected = control_from_physics_profile(profile)
    if control is None:
        return expected
    mismatches = [
        name
        for name in (
            "mass_kg",
            "gravity",
            "linear_drag",
            "thrust_scale",
            "rate_tau_s",
            "motor_tau_s",
        )
        if not math.isclose(getattr(control, name), getattr(expected, name))
    ]
    if not np.allclose(control.max_rate_rad_s, expected.max_rate_rad_s):
        mismatches.append("max_rate_rad_s")
    if mismatches:
        raise ValueError(
            "MuJoCo control conflicts with physics_profile: "
            + ", ".join(mismatches)
        )
    return control


def step_actuator_targets(
    thrust_state: float,
    rate_state: np.ndarray,
    action: np.ndarray,
    control: MuJoCoControlParams,
    dt: float,
) -> tuple[float, np.ndarray]:
    target_thrust = 1.0 + control.thrust_scale * float(action[0])
    thrust = thrust_state + first_order_alpha(dt, control.motor_tau_s) * (
        target_thrust - thrust_state
    )
    target_rates = np.asarray(action[1:4], dtype=np.float64) * np.asarray(
        control.max_rate_rad_s,
        dtype=np.float64,
    )
    rates = np.asarray(rate_state, dtype=np.float64) + first_order_alpha(
        dt,
        control.rate_tau_s,
    ) * (target_rates - rate_state)
    return float(thrust), rates


def rate_control_torque(
    target_rates: np.ndarray,
    current_rates: np.ndarray,
    control: MuJoCoControlParams,
) -> np.ndarray:
    error = np.asarray(target_rates, dtype=np.float64) - np.asarray(
        current_rates,
        dtype=np.float64,
    )
    # Damping is relative to the commanded rate so it cannot bias the
    # zero-torque equilibrium away from that command.
    return (control.rate_kp + control.rate_kd) * error


def apply_control(env, idx: int, data, action: np.ndarray) -> None:
    data.xfrc_applied[:] = 0.0
    rotation = np.asarray(data.xmat[env.body_id], dtype=np.float64).reshape(3, 3)
    current_rates = np.asarray(data.qvel[3:6], dtype=np.float64)
    env.thrust_state[idx], env.rate_command_state[idx] = step_actuator_targets(
        env.thrust_state[idx],
        env.rate_command_state[idx],
        action,
        env.control,
        env.dt,
    )
    thrust = env.control.mass_kg * env.control.gravity * env.thrust_state[idx]
    torque_body = rate_control_torque(
        env.rate_command_state[idx],
        current_rates,
        env.control,
    )
    data.xfrc_applied[env.body_id, :3] = rotation[:, 2] * thrust
    data.xfrc_applied[env.body_id, :3] -= (
        env.control.linear_drag
        * env.control.mass_kg
        * np.asarray(data.qvel[:3], dtype=np.float64)
    )
    disturbance = disturbance_accel(env)
    if disturbance is not None:
        data.xfrc_applied[env.body_id, :3] += disturbance[idx] * env.control.mass_kg
    data.xfrc_applied[env.body_id, 3:] = rotation @ torque_body


def first_order_alpha(dt: float, tau_s: float) -> float:
    dt = require_finite_real(
        dt,
        "MuJoCo timestep",
        minimum=0.0,
        strictly_greater=True,
    )
    tau_s = require_finite_real(
        tau_s,
        "actuator time constant",
        minimum=0.0,
    )
    return 1.0 if tau_s == 0.0 else dt / (tau_s + dt)
