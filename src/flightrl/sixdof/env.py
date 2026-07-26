from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curriculum import ResetProfile, resolve_reset_profile, sample_initial_velocity, sample_reset
from .dynamics import step_body_rate
from .geometry import BoxRoom, body_rays_world, normalize_quat
from .motor_rpm import MotorRpmParams, resolve_motor_rpm_params, step_motor_rpm
from .native_reset import native_reset_one
from .physics import (
    MAX_RATE_PITCH,
    MAX_RATE_ROLL,
    MAX_RATE_YAW,
    SixDofDomainRandomization,
    SixDofPhysicsProfile,
    resolve_domain_randomization,
    resolve_physics_profile,
    sample_physics,
)
from .sensor_model import SixDofSensorProfile, noisy_values, observed_ranges, resolve_sensor_profile


OBSERVATION_DIM = 28
ACTION_DIM = 4
TASK_IDS = {"position_yaw": 0, "obstacle_avoidance": 1, "attitude": 2, "circle": 3}
REWARD_MODE_IDS = {"env": 0, "progress": 1, "progress_clearance": 2, "progress_yaw_clearance": 3, "live_clearance": 4, "live_stable_clearance": 5}


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
        reset_profile: str | ResetProfile | None = None,
        action_mode: str = "body_rate",
        physics_profile: str | SixDofPhysicsProfile | None = None,
        motor_rpm_profile: str | MotorRpmParams | None = None,
        domain_randomization: str | SixDofDomainRandomization | None = None,
        sensor_profile: str | SixDofSensorProfile | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.task = task
        if action_mode not in {"body_rate", "motor_rpm"}:
            raise ValueError(f"unknown 6-DoF action mode {action_mode!r}")
        self.action_mode = action_mode
        self.hardware_action_interface = "sim_only_motor_rpm" if self.action_mode == "motor_rpm" else "firmware_setpoint"
        self.use_native_step = bool(use_native_step)
        self.reset_profile = resolve_reset_profile(reset_profile)
        self.physics_profile = resolve_physics_profile(physics_profile)
        self.domain_randomization = resolve_domain_randomization(domain_randomization)
        self.sensor_profile = resolve_sensor_profile(sensor_profile)
        self.teacher_profile = "default"
        self.room = room or BoxRoom()
        self.room_bounds = np.asarray([self.room.x_min, self.room.x_max, self.room.y_min, self.room.y_max, self.room.z_min, self.room.z_max, self.room.max_range_m], dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.mass, self.gravity, self.drag = self.physics_profile.mass_kg, self.physics_profile.gravity_m_s2, self.physics_profile.linear_drag
        self.max_rate = np.asarray(self.physics_profile.max_rate_rad_s, dtype=np.float32)
        self.position = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.velocity = np.zeros_like(self.position)
        self.quaternion = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.body_rates = np.zeros_like(self.position)
        self.physics_params = np.repeat(self.physics_profile.as_array()[None, :], self.num_envs, axis=0).astype(np.float32)
        self.thrust_state = np.ones(self.num_envs, dtype=np.float32)
        self.command_state = np.zeros((self.num_envs, ACTION_DIM), dtype=np.float32)
        self.previous_action = np.zeros((self.num_envs, ACTION_DIM), dtype=np.float32)
        self.motor_params = resolve_motor_rpm_params(motor_rpm_profile or ("puffer_drone" if physics_profile == "puffer_drone" else None))
        self.motor_hover_rpm = self.motor_params.hover_rpm
        self.motor_rpm = np.zeros((self.num_envs, ACTION_DIM), dtype=np.float32)
        self.target_position = np.zeros_like(self.position)
        self.target_yaw = np.zeros(self.num_envs, dtype=np.float32)
        self.ranges_m = np.zeros((self.num_envs, 6), dtype=np.float32)
        self.step_count = np.zeros(self.num_envs, dtype=np.int32)
        self.observations = np.zeros((self.num_envs, OBSERVATION_DIM), dtype=np.float32)
        self.rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.terminals = np.zeros(self.num_envs, dtype=np.uint8)
        self.truncations = np.zeros(self.num_envs, dtype=np.uint8)
        self.native_task_ids = np.full(self.num_envs, TASK_IDS.get(self.task, 0), dtype=np.int32)
        self.native_reward_mode_id = 0
        self.native_previous_error = np.zeros(self.num_envs, dtype=np.float32)
        self.native_context_required = self.task == "circle"
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, list[dict[str, float]]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_mask(np.ones(self.num_envs, dtype=bool))
        self._update_ranges()
        self.observations[:] = self.observation()
        return self.observations, []

    def reset_done(self, done: np.ndarray) -> np.ndarray:
        mask = np.asarray(done, dtype=bool)
        if np.any(mask):
            self._reset_mask(mask)
            self._update_ranges()
            self.observations[:] = self.observation()
        return self.observations

    def step(self, actions: np.ndarray):
        clipped = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        executed = self._executed_action(clipped)
        if self.use_native_step and self.action_mode != "motor_rpm":
            from .native import native_step_env

            native_step_env(self, executed)
            if self.sensor_profile.observation_enabled:
                self.observations[:] = self.observation()
            return self.observations, self.rewards, self.terminals, self.truncations, []
        elif self.action_mode == "motor_rpm":
            step_motor_rpm(self, executed)
        else:
            self._python_step(executed)
        self.step_count += 1
        reward = self._reward(executed)
        terminated = ~self.room.contains(self.position)
        truncated = self.step_count >= 800
        self.previous_action[:] = executed
        self.observations[:] = self.observation()
        self.rewards[:] = reward
        self.terminals[:] = terminated.astype(np.uint8)
        self.truncations[:] = truncated.astype(np.uint8)
        return self.observations, self.rewards, self.terminals, self.truncations, []

    def _python_step(self, clipped: np.ndarray) -> None:
        step_body_rate(self, clipped)

    def observation(self) -> np.ndarray:
        position = noisy_values(self.position, self.sensor_profile.state_noise_std_m, self.rng)
        velocity = noisy_values(self.velocity, self.sensor_profile.velocity_noise_std_m_s, self.rng)
        body_rates = noisy_values(self.body_rates, self.sensor_profile.body_rate_noise_std_rad_s, self.rng)
        ranges = observed_ranges(self.ranges_m, max_range_m=self.room.max_range_m, profile=self.sensor_profile, rng=self.rng)
        pos_error = self.target_position - position
        yaw = quat_to_yaw(self.quaternion)
        yaw_error = wrap_angle(self.observation_target_yaw() - yaw)
        obs = np.concatenate(
            [
                pos_error / np.asarray([2.0, 2.0, 1.5], dtype=np.float32),
                velocity / 3.0,
                self.quaternion,
                body_rates / self.physics_params[:, [MAX_RATE_ROLL, MAX_RATE_PITCH, MAX_RATE_YAW]],
                self.target_position / np.asarray([2.0, 2.0, 2.5], dtype=np.float32),
                np.sin(yaw_error)[:, None],
                np.cos(yaw_error)[:, None],
                ranges / self.room.max_range_m,
                self.previous_action,
            ],
            axis=1,
        )
        return obs.astype(np.float32)

    def observation_target_yaw(self) -> np.ndarray:
        if self.task != "circle":
            return self.target_yaw
        from .yaw import circle_tangent_yaw

        return circle_tangent_yaw(self)

    def native_reset(self, seed: int = 0) -> np.ndarray:
        rng = np.uint32(seed)
        for idx in range(self.num_envs):
            rng = native_reset_one(self, idx, rng)
        self._update_ranges()
        self.observations[:] = self.observation()
        return self.observations

    def set_native_context(
        self,
        *,
        task_indices: np.ndarray | None = None,
        tasks: tuple[str, ...] | None = None,
        reward_mode: str = "env",
        previous_error: np.ndarray | None = None,
    ) -> None:
        if reward_mode not in REWARD_MODE_IDS:
            raise ValueError(f"unknown native reward mode {reward_mode!r}")
        self.native_reward_mode_id = REWARD_MODE_IDS[reward_mode]
        if task_indices is None:
            self.native_task_ids.fill(TASK_IDS.get(self.task, 0))
        else:
            task_names = tasks or (self.task,)
            ids = np.asarray([TASK_IDS[name] for name in task_names], dtype=np.int32)
            self.native_task_ids[:] = ids[np.asarray(task_indices, dtype=np.int64)]
        if previous_error is None:
            self.native_previous_error.fill(0.0)
        else:
            self.native_previous_error[:] = np.asarray(previous_error, dtype=np.float32)
        self.native_context_required = self.native_reward_mode_id != 0 or bool(np.any(self.native_task_ids == TASK_IDS["circle"]))

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

    def _reset_mask(self, mask: np.ndarray) -> None:
        count = int(np.sum(mask))
        if count == 0:
            return
        position, roll, pitch, yaw, target, target_yaw = sample_reset(self.reset_profile, self.rng, count, self.room)
        self.position[mask] = position
        self.velocity[mask] = sample_initial_velocity(self.reset_profile, self.rng, count)
        self.body_rates[mask] = 0.0
        self.quaternion[mask] = euler_to_quat(roll, pitch, yaw)
        self.physics_params[mask] = sample_physics(self.physics_profile, self.domain_randomization, self.rng, count)
        self.thrust_state[mask] = 1.0
        self.command_state[mask] = 0.0
        self.target_position[mask] = target
        self.target_yaw[mask] = target_yaw
        self.previous_action[mask] = 0.0
        self.motor_rpm[mask] = self.motor_hover_rpm
        self.step_count[mask] = 0
        self.rewards[mask] = 0.0
        self.terminals[mask] = 0
        self.truncations[mask] = 0

    def _executed_action(self, clipped: np.ndarray) -> np.ndarray:
        alpha = self.sensor_profile.action_alpha(self.dt)
        if alpha >= 1.0:
            self.command_state[:] = clipped
        else:
            self.command_state += alpha * (clipped - self.command_state)
        return self.command_state.astype(np.float32)

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
