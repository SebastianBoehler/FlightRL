from __future__ import annotations

import numpy as np

from .disturbance import disturbance_accel
from .geometry import quat_to_matrix
from .physics import (
    GRAVITY,
    LINEAR_DRAG,
    MASS,
    MAX_RATE_PITCH,
    MAX_RATE_ROLL,
    MAX_RATE_YAW,
    MOTOR_TAU,
    RATE_TAU,
    THRUST_SCALE,
)


def step_body_rate(env, clipped: np.ndarray) -> None:
    params = env.physics_params
    target_thrust = 1.0 + params[:, THRUST_SCALE] * clipped[:, 0]
    motor_tau = params[:, MOTOR_TAU]
    motor_alpha = np.where(motor_tau > 0.0, env.dt / (motor_tau + env.dt), 1.0).astype(np.float32)
    env.thrust_state += motor_alpha * (target_thrust - env.thrust_state)
    thrust = params[:, MASS] * params[:, GRAVITY] * env.thrust_state
    max_rates = params[:, [MAX_RATE_ROLL, MAX_RATE_PITCH, MAX_RATE_YAW]]
    commanded_rates = clipped[:, 1:4] * max_rates
    alpha = env.dt / (params[:, RATE_TAU][:, None] + env.dt)
    env.body_rates += alpha * (commanded_rates - env.body_rates)
    env._integrate_orientation()
    up_world = quat_to_matrix(env.quaternion)[:, :, 2]
    acceleration = up_world * (thrust / params[:, MASS])[:, None]
    acceleration[:, 2] -= params[:, GRAVITY]
    acceleration -= params[:, LINEAR_DRAG][:, None] * env.velocity
    disturbance = disturbance_accel(env)
    if disturbance is not None:
        acceleration += disturbance
    env.velocity += acceleration.astype(np.float32) * env.dt
    env.position += env.velocity * env.dt
    env._update_ranges()
