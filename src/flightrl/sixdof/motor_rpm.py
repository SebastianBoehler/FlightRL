from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import normalize_quat, quat_to_matrix


@dataclass(frozen=True, slots=True)
class MotorRpmParams:
    hover_rpm: float = 16000.0
    max_rpm: float = 32000.0
    motor_tau_s: float = 0.035
    arm_length_m: float = 0.046
    ixx: float = 1.7e-5
    iyy: float = 1.7e-5
    izz: float = 2.9e-5
    yaw_torque_gain: float = 0.018
    angular_drag: float = 0.0025


def step_motor_rpm(env, actions: np.ndarray) -> None:
    params = env.motor_params
    target = target_rpm(actions, params)
    alpha = env.dt / (params.motor_tau_s + env.dt)
    env.motor_rpm += alpha * (target - env.motor_rpm)
    thrusts = thrust_from_rpm(env.motor_rpm, env.mass, env.gravity, params)
    integrate_state(env, thrusts, params)
    env.previous_action[:] = actions
    env._update_ranges()


def target_rpm(actions: np.ndarray, params: MotorRpmParams) -> np.ndarray:
    min_rpm = max(0.0, 2.0 * params.hover_rpm - params.max_rpm)
    unit = (np.clip(actions, -1.0, 1.0) + 1.0) * 0.5
    return (min_rpm + unit * (params.max_rpm - min_rpm)).astype(np.float32)


def thrust_from_rpm(rpm: np.ndarray, mass: float, gravity: float, params: MotorRpmParams) -> np.ndarray:
    coeff = (mass * gravity) / (4.0 * params.hover_rpm * params.hover_rpm)
    return (coeff * np.maximum(rpm, 0.0) ** 2).astype(np.float32)


def integrate_state(env, thrusts: np.ndarray, params: MotorRpmParams) -> None:
    total = np.sum(thrusts, axis=1)
    torque = motor_torques(thrusts, params)
    inertia = np.asarray([params.ixx, params.iyy, params.izz], dtype=np.float32)
    env.body_rates += ((torque - params.angular_drag * env.body_rates) / inertia) * env.dt
    env.body_rates = np.clip(env.body_rates, -env.max_rate, env.max_rate).astype(np.float32)
    integrate_orientation(env)
    up_world = quat_to_matrix(env.quaternion)[:, :, 2]
    acceleration = up_world * (total / env.mass)[:, None]
    acceleration[:, 2] -= env.gravity
    acceleration -= env.drag * env.velocity
    env.velocity += acceleration.astype(np.float32) * env.dt
    env.position += env.velocity * env.dt


def motor_torques(thrusts: np.ndarray, params: MotorRpmParams) -> np.ndarray:
    arm = params.arm_length_m / np.sqrt(2.0)
    torque = np.zeros((thrusts.shape[0], 3), dtype=np.float32)
    torque[:, 0] = arm * ((thrusts[:, 2] + thrusts[:, 3]) - (thrusts[:, 0] + thrusts[:, 1]))
    torque[:, 1] = arm * ((thrusts[:, 1] + thrusts[:, 2]) - (thrusts[:, 0] + thrusts[:, 3]))
    torque[:, 2] = params.yaw_torque_gain * (-thrusts[:, 0] + thrusts[:, 1] - thrusts[:, 2] + thrusts[:, 3])
    return torque


def integrate_orientation(env) -> None:
    q = env.quaternion
    omega = env.body_rates
    omega_quat = np.concatenate([np.zeros((env.num_envs, 1), dtype=np.float32), omega], axis=1)
    env.quaternion = normalize_quat(q + 0.5 * quat_mul(q, omega_quat) * env.dt).astype(np.float32)


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left[:, 0], left[:, 1], left[:, 2], left[:, 3]
    rw, rx, ry, rz = right[:, 0], right[:, 1], right[:, 2], right[:, 3]
    return np.stack(
        [lw * rw - lx * rx - ly * ry - lz * rz, lw * rx + lx * rw + ly * rz - lz * ry, lw * ry - lx * rz + ly * rw + lz * rx, lw * rz + lx * ry - ly * rx + lz * rw],
        axis=1,
    ).astype(np.float32)
