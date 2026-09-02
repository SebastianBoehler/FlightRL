from __future__ import annotations

import numpy as np

from .curriculum import ResetProfile, resolve_reset_profile, sample_initial_velocity, sample_reset
from .dynamics import step_body_rate
from .embodiment import embodiment_batch
from .geometry import BoxRoom, body_rays_world, normalize_quat
from .motor_rpm import MotorRpmParams, resolve_motor_rpm_params, step_motor_rpm
from .native_reset import native_reset_one
from .orientation import euler_to_quat, quat_mul, quat_to_yaw
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
from .snapshot import SixDofSnapshot
from .task_context import CIRCLE_TASK_ID, default_task_reward, task_target_yaw, wrap_angle, write_yaw_observation
from .validation import (
    action_batch,
    finite_batch,
    require_bool,
    require_choice,
    require_finite_real,
    require_positive_int,
    reset_mask,
    task_id_batch,
)


OBSERVATION_DIM = 28
ACTION_DIM = 4
TASK_IDS = {
    "position_yaw": 0,
    "obstacle_avoidance": 1,
    "circle": CIRCLE_TASK_ID,
}
REWARD_MODE_IDS = {"env": 0, "progress": 1, "progress_clearance": 2, "progress_yaw_clearance": 3, "live_clearance": 4, "live_stable_clearance": 5}

class SixDofEnv:
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
        self.num_envs = require_positive_int(num_envs, "num_envs")
        self.dt = require_finite_real(
            dt,
            "dt",
            minimum=0.0,
            strictly_greater=True,
        )
        self.task = require_choice(task, "6-DoF task", TASK_IDS)
        self.action_mode = require_choice(
            action_mode,
            "6-DoF action mode",
            {"body_rate", "motor_rpm"},
        )
        self.hardware_action_interface = "sim_only_motor_rpm" if self.action_mode == "motor_rpm" else "firmware_setpoint"
        self.use_native_step = require_bool(use_native_step, "use_native_step")
        if self.use_native_step and self.action_mode != "body_rate":
            raise ValueError("native 6-DoF stepping supports only body_rate actions")
        self.reset_profile = resolve_reset_profile(reset_profile)
        self.physics_profile = resolve_physics_profile(physics_profile)
        self.domain_randomization = resolve_domain_randomization(domain_randomization)
        self.sensor_profile = resolve_sensor_profile(sensor_profile)
        self.teacher_profile = "default"
        if room is not None and not isinstance(room, BoxRoom):
            raise TypeError("room must be a BoxRoom")
        self.room = room or BoxRoom()
        if self.use_native_step and self.room.obstacles:
            raise ValueError("native 6-DoF stepping does not support BoxRoom interior obstacles")
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
        self.motor_params = resolve_motor_rpm_params(
            motor_rpm_profile
            or ("puffer_parameters" if physics_profile == "puffer_parameters" else None)
        )
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
        self.native_task_ids = np.full(self.num_envs, TASK_IDS[self.task], dtype=np.int32)
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
        mask = reset_mask(done, self.num_envs)
        if np.any(mask):
            self._reset_mask(mask)
            self._update_ranges()
            self.observations[:] = self.observation()
        return self.observations

    def step(self, actions: np.ndarray):
        clipped = action_batch(actions, self.num_envs, ACTION_DIM)
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
        return task_target_yaw(
            self.position,
            self.target_position,
            self.target_yaw,
            self.native_task_ids,
        )

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
            self.native_task_ids.fill(TASK_IDS[self.task])
        else:
            task_names = tasks or (self.task,)
            self.native_task_ids[:] = task_id_batch(
                task_indices,
                task_names,
                num_envs=self.num_envs,
                task_ids=TASK_IDS,
            )
        if previous_error is None:
            self.native_previous_error.fill(0.0)
        else:
            self.native_previous_error[:] = finite_batch(
                previous_error,
                "previous error",
                self.num_envs,
            )
        self.native_context_required = self.native_reward_mode_id != 0 or bool(np.any(self.native_task_ids == TASK_IDS["circle"]))
        write_yaw_observation(self.observations, self.observation_target_yaw(), quat_to_yaw(self.quaternion))

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

    def embodiment_descriptors(self) -> np.ndarray:
        """Return the physical descriptor for each sampled environment."""
        return embodiment_batch(self.physics_params)

    def _reset_mask(self, mask: np.ndarray) -> None:
        count = int(np.sum(mask))
        if count == 0:
            return
        position, roll, pitch, yaw, target, target_yaw = sample_reset(self.reset_profile, self.rng, count, self.room)
        self.position[mask] = position
        self.velocity[mask] = sample_initial_velocity(self.reset_profile, self.rng, count)
        self.body_rates[mask] = 0.0
        self.quaternion[mask] = euler_to_quat(roll, pitch, yaw)
        self.physics_params[mask] = sample_physics(
            self.physics_profile,
            self.domain_randomization,
            self.rng,
            count,
            action_mode=self.action_mode,
        )
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
        positions = np.repeat(self.position, rays.shape[1], axis=0)
        self.ranges_m[:] = self.room.raycast(
            positions,
            rays.reshape(-1, 3),
        ).reshape(self.num_envs, rays.shape[1])

    def _integrate_orientation(self) -> None:
        omega = self.body_rates
        q = self.quaternion
        omega_quat = np.concatenate([np.zeros((self.num_envs, 1), dtype=np.float32), omega], axis=1)
        q_dot = 0.5 * quat_mul(q, omega_quat)
        self.quaternion = normalize_quat(q + q_dot * self.dt).astype(np.float32)

    def _reward(self, actions: np.ndarray) -> np.ndarray:
        return default_task_reward(self, actions, quat_to_yaw(self.quaternion))
