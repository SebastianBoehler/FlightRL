from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import BoxRoom, body_rays_world, normalize_quat, quat_to_matrix


OBSERVATION_DIM = 28
ACTION_DIM = 4


@dataclass(slots=True)
class SixDofSnapshot:
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    body_rates: np.ndarray
    target_position: np.ndarray
    target_yaw: np.ndarray
    ranges_m: np.ndarray


class SixDofCrazyflieEnv:
    def __init__(
        self,
        num_envs: int = 256,
        seed: int = 0,
        room: BoxRoom | None = None,
        dt: float = 0.01,
        task: str = "position_yaw",
        use_native_step: bool = False,
    ) -> None:
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.task = task
        self.use_native_step = bool(use_native_step)
        self.room = room or BoxRoom()
        self.rng = np.random.default_rng(seed)
        self.mass = 0.036
        self.gravity = 9.81
        self.drag = 0.10
        self.rate_tau = 0.045
        self.max_rate = np.asarray([6.0, 6.0, 4.0], dtype=np.float32)
        self.position = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.velocity = np.zeros_like(self.position)
        self.quaternion = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.body_rates = np.zeros_like(self.position)
        self.previous_action = np.zeros((self.num_envs, ACTION_DIM), dtype=np.float32)
        self.target_position = np.zeros_like(self.position)
        self.target_yaw = np.zeros(self.num_envs, dtype=np.float32)
        self.ranges_m = np.zeros((self.num_envs, 6), dtype=np.float32)
        self.step_count = np.zeros(self.num_envs, dtype=np.int32)
        self.observations = np.zeros((self.num_envs, OBSERVATION_DIM), dtype=np.float32)
        self.rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.terminals = np.zeros(self.num_envs, dtype=np.uint8)
        self.truncations = np.zeros(self.num_envs, dtype=np.uint8)
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, list[dict[str, float]]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.position[:, 0] = self.rng.uniform(-0.8, 0.8, self.num_envs)
        self.position[:, 1] = self.rng.uniform(-0.8, 0.8, self.num_envs)
        self.position[:, 2] = self.rng.uniform(0.35, 0.9, self.num_envs)
        self.velocity.fill(0.0)
        self.body_rates.fill(0.0)
        self.quaternion[:] = euler_to_quat(
            self.rng.normal(0.0, 0.08, self.num_envs),
            self.rng.normal(0.0, 0.08, self.num_envs),
            self.rng.uniform(-np.pi, np.pi, self.num_envs),
        )
        self.target_position[:, 0] = self.rng.uniform(-1.0, 1.0, self.num_envs)
        self.target_position[:, 1] = self.rng.uniform(-1.0, 1.0, self.num_envs)
        self.target_position[:, 2] = self.rng.uniform(0.45, 0.9, self.num_envs)
        self.target_yaw[:] = self.rng.uniform(-np.pi, np.pi, self.num_envs)
        self.previous_action.fill(0.0)
        self.step_count.fill(0)
        self.rewards.fill(0.0)
        self.terminals.fill(0)
        self.truncations.fill(0)
        self._update_ranges()
        self.observations[:] = self.observation()
        return self.observations, []

    def step(self, actions: np.ndarray):
        clipped = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        if self.use_native_step:
            from .native import native_step_env

            native_step_env(self, clipped)
            return self.observations, self.rewards, self.terminals, self.truncations, []
        else:
            self._python_step(clipped)
        self.step_count += 1
        reward = self._reward(clipped)
        terminated = ~self.room.contains(self.position)
        truncated = self.step_count >= 800
        self.previous_action[:] = clipped
        self.observations[:] = self.observation()
        self.rewards[:] = reward
        self.terminals[:] = terminated.astype(np.uint8)
        self.truncations[:] = truncated.astype(np.uint8)
        return self.observations, self.rewards, self.terminals, self.truncations, []

    def _python_step(self, clipped: np.ndarray) -> None:
        thrust = self.mass * self.gravity * (1.0 + 0.75 * clipped[:, 0])
        commanded_rates = clipped[:, 1:4] * self.max_rate
        alpha = self.dt / (self.rate_tau + self.dt)
        self.body_rates += alpha * (commanded_rates - self.body_rates)
        self._integrate_orientation()
        up_world = quat_to_matrix(self.quaternion)[:, :, 2]
        acceleration = up_world * (thrust / self.mass)[:, None]
        acceleration[:, 2] -= self.gravity
        acceleration -= self.drag * self.velocity
        self.velocity += acceleration.astype(np.float32) * self.dt
        self.position += self.velocity * self.dt
        self._update_ranges()

    def observation(self) -> np.ndarray:
        pos_error = self.target_position - self.position
        yaw = quat_to_yaw(self.quaternion)
        yaw_error = wrap_angle(self.target_yaw - yaw)
        obs = np.concatenate(
            [
                pos_error / np.asarray([2.0, 2.0, 1.5], dtype=np.float32),
                self.velocity / 3.0,
                self.quaternion,
                self.body_rates / self.max_rate,
                self.target_position / np.asarray([2.0, 2.0, 2.5], dtype=np.float32),
                np.sin(yaw_error)[:, None],
                np.cos(yaw_error)[:, None],
                self.ranges_m / self.room.max_range_m,
                self.previous_action,
            ],
            axis=1,
        )
        return obs.astype(np.float32)

    def snapshot(self) -> SixDofSnapshot:
        return SixDofSnapshot(
            self.position.copy(),
            self.velocity.copy(),
            self.quaternion.copy(),
            self.body_rates.copy(),
            self.target_position.copy(),
            self.target_yaw.copy(),
            self.ranges_m.copy(),
        )

    def _update_ranges(self) -> None:
        rays = body_rays_world(self.quaternion)
        for sensor_idx in range(6):
            self.ranges_m[:, sensor_idx] = self.room.raycast(self.position, rays[:, sensor_idx, :])

    def _integrate_orientation(self) -> None:
        omega = self.body_rates
        q = self.quaternion
        omega_quat = np.concatenate([np.zeros((self.num_envs, 1), dtype=np.float32), omega], axis=1)
        q_dot = 0.5 * quat_mul(q, omega_quat)
        self.quaternion = normalize_quat(q + q_dot * self.dt).astype(np.float32)

    def _reward(self, actions: np.ndarray) -> np.ndarray:
        pos_error = np.linalg.norm(self.target_position - self.position, axis=1)
        speed = np.linalg.norm(self.velocity, axis=1)
        yaw_error = np.abs(wrap_angle(self.target_yaw - quat_to_yaw(self.quaternion)))
        clearance_penalty = np.maximum(0.0, 0.35 - np.min(self.ranges_m[:, :4], axis=1))
        control = np.linalg.norm(actions, axis=1)
        return (1.0 - pos_error - 0.15 * speed - 0.1 * yaw_error - 1.5 * clearance_penalty - 0.02 * control).astype(np.float32)


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left[:, 0], left[:, 1], left[:, 2], left[:, 3]
    rw, rx, ry, rz = right[:, 0], right[:, 1], right[:, 2], right[:, 3]
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=1,
    ).astype(np.float32)


def euler_to_quat(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.stack(
        [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy],
        axis=1,
    ).astype(np.float32)


def quat_to_yaw(quaternions: np.ndarray) -> np.ndarray:
    q = normalize_quat(quaternions)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(np.float32)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
