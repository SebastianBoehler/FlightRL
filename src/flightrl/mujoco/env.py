from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flightrl.mujoco.model import load_crazyflie_model, require_mujoco
from flightrl.mujoco.semantic_scene import box_room_from_semantic_scene
from flightrl.navigation.semantic_scene import SemanticScene
from flightrl.sixdof.curriculum import ResetProfile, resolve_reset_profile, sample_reset
from flightrl.sixdof.disturbance import disturbance_accel
from flightrl.sixdof.env import ACTION_DIM, OBSERVATION_DIM, euler_to_quat, quat_to_yaw, wrap_angle
from flightrl.sixdof.geometry import BoxRoom, body_rays_world
from flightrl.sixdof.physics import SixDofPhysicsProfile, resolve_physics_profile
from flightrl.sixdof.sensor_model import SixDofSensorProfile, noisy_values, observed_ranges, resolve_sensor_profile


@dataclass(frozen=True, slots=True)
class MuJoCoControlParams:
    mass_kg: float = 0.036
    gravity: float = 9.81
    max_rate_rad_s: tuple[float, float, float] = (6.0, 6.0, 4.0)
    rate_kp: float = 2.4e-4
    rate_kd: float = 3.0e-5
    thrust_scale: float = 0.75


class MuJoCoCrazyflieEnv:
    """MuJoCo-backed 6-DoF Crazyflie test backend.

    This backend keeps the FlightRL observation/action/task contract while using
    MuJoCo for rigid-body integration and contacts. It is intentionally not a
    vectorized Ocean replacement.
    """

    def __init__(
        self,
        num_envs: int = 1,
        seed: int = 0,
        room: BoxRoom | None = None,
        semantic_scene: SemanticScene | None = None,
        dt: float = 0.01,
        task: str = "position_yaw",
        reset_profile: str | ResetProfile | None = None,
        control: MuJoCoControlParams | None = None,
        sensor_profile: str | SixDofSensorProfile | None = None,
        physics_profile: str | SixDofPhysicsProfile | None = None,
    ) -> None:
        self.mujoco = require_mujoco()
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.task = task
        self.reset_profile = resolve_reset_profile(reset_profile)
        self.sensor_profile = resolve_sensor_profile(sensor_profile)
        self.physics_profile = resolve_physics_profile(physics_profile)
        if room is not None and semantic_scene is not None:
            raise ValueError("provide either room or semantic_scene, not both")
        self.semantic_scene = semantic_scene
        self.room = (
            box_room_from_semantic_scene(semantic_scene)
            if semantic_scene is not None
            else room or BoxRoom()
        )
        self.rng = np.random.default_rng(seed)
        self.control = control or control_from_physics_profile(self.physics_profile)
        self.gravity = float(self.control.gravity)
        self.max_rate = np.asarray(self.control.max_rate_rad_s, dtype=np.float32)

        self.model = load_crazyflie_model(self.dt, scene=semantic_scene)
        self.body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "crazyflie")
        if self.body_id < 0:
            raise RuntimeError("MuJoCo Crazyflie model is missing body 'crazyflie'")
        self.data = [self.mujoco.MjData(self.model) for _ in range(self.num_envs)]

        self.position = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.velocity = np.zeros_like(self.position)
        self.quaternion = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.body_rates = np.zeros_like(self.position)
        self.command_state = np.zeros((self.num_envs, ACTION_DIM), dtype=np.float32)
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
        self._reset_mask(np.ones(self.num_envs, dtype=bool))
        self._sync_state_from_data()
        self._update_ranges()
        self.observations[:] = self.observation()
        return self.observations, []

    def reset_done(self, done: np.ndarray) -> np.ndarray:
        mask = np.asarray(done, dtype=bool)
        if np.any(mask):
            self._reset_mask(mask)
            self._sync_state_from_data()
            self._update_ranges()
            self.observations[:] = self.observation()
        return self.observations

    def step(self, actions: np.ndarray):
        clipped = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        executed = self._executed_action(clipped)
        for idx, data in enumerate(self.data):
            self._apply_control(idx, data, executed[idx])
            self.mujoco.mj_step(self.model, data)
        self._sync_state_from_data()
        self._update_ranges()
        self.step_count += 1
        self.rewards[:] = self._reward(executed)
        self.terminals[:] = (~self.room.contains(self.position)).astype(np.uint8)
        self.truncations[:] = (self.step_count >= 800).astype(np.uint8)
        self.previous_action[:] = executed
        self.observations[:] = self.observation()
        return self.observations, self.rewards, self.terminals, self.truncations, []

    def observation(self) -> np.ndarray:
        position = noisy_values(self.position, self.sensor_profile.state_noise_std_m, self.rng)
        velocity = noisy_values(self.velocity, self.sensor_profile.velocity_noise_std_m_s, self.rng)
        body_rates = noisy_values(self.body_rates, self.sensor_profile.body_rate_noise_std_rad_s, self.rng)
        ranges = observed_ranges(self.ranges_m, max_range_m=self.room.max_range_m, profile=self.sensor_profile, rng=self.rng)
        pos_error = self.target_position - position
        yaw_error = wrap_angle(self.target_yaw - quat_to_yaw(self.quaternion))
        obs = np.concatenate(
            [
                pos_error / np.asarray([2.0, 2.0, 1.5], dtype=np.float32),
                velocity / 3.0,
                self.quaternion,
                body_rates / self.max_rate,
                self.target_position / np.asarray([2.0, 2.0, 2.5], dtype=np.float32),
                np.sin(yaw_error)[:, None],
                np.cos(yaw_error)[:, None],
                ranges / self.room.max_range_m,
                self.previous_action,
            ],
            axis=1,
        )
        return obs.astype(np.float32)

    def render_rgb(
        self,
        width: int = 640,
        height: int = 480,
        env_index: int = 0,
        *,
        camera: str | None = None,
    ) -> np.ndarray:
        with self.mujoco.Renderer(self.model, height=height, width=width) as renderer:
            renderer.update_scene(self.data[env_index], camera=camera)
            return renderer.render()

    def render_aideck_gray4(self, width: int = 64, height: int = 48, env_index: int = 0) -> np.ndarray:
        gray = self.render_aideck_gray(width, height, env_index)
        quantized = np.rint(gray.astype(np.float32) / 17.0) * 17.0
        return np.clip(quantized, 0.0, 255.0).astype(np.uint8)

    def render_aideck_gray(self, width: int = 64, height: int = 48, env_index: int = 0) -> np.ndarray:
        rgb = self.render_rgb(width, height, env_index, camera="aideck").astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return np.clip(gray, 0.0, 255.0).astype(np.uint8)

    def _reset_mask(self, mask: np.ndarray) -> None:
        count = int(np.sum(mask))
        if count == 0:
            return
        positions, roll, pitch, yaw, targets, target_yaw = sample_reset(self.reset_profile, self.rng, count, self.room)
        quaternions = euler_to_quat(roll, pitch, yaw)
        row = 0
        for idx, reset in enumerate(mask):
            if not reset:
                continue
            data = self.data[idx]
            self.mujoco.mj_resetData(self.model, data)
            data.qpos[:3] = positions[row]
            data.qpos[3:7] = quaternions[row]
            data.qvel[:] = 0.0
            self.mujoco.mj_forward(self.model, data)
            self.target_position[idx] = targets[row]
            self.target_yaw[idx] = target_yaw[row]
            row += 1
        self.command_state[mask] = 0.0
        self.previous_action[mask] = 0.0
        self.step_count[mask] = 0
        self.rewards[mask] = 0.0
        self.terminals[mask] = 0
        self.truncations[mask] = 0

    def _apply_control(self, idx: int, data, action: np.ndarray) -> None:
        data.xfrc_applied[:] = 0.0
        rotation = np.asarray(data.xmat[self.body_id], dtype=np.float64).reshape(3, 3)
        current_rates = np.asarray(data.qvel[3:6], dtype=np.float64)
        target_rates = action[1:4].astype(np.float64) * self.max_rate.astype(np.float64)
        thrust = self.control.mass_kg * self.control.gravity * (1.0 + self.control.thrust_scale * float(action[0]))
        torque_body = self.control.rate_kp * (target_rates - current_rates) - self.control.rate_kd * current_rates
        data.xfrc_applied[self.body_id, :3] = rotation[:, 2] * thrust
        disturbance = disturbance_accel(self)
        if disturbance is not None:
            data.xfrc_applied[self.body_id, :3] += disturbance[idx] * self.control.mass_kg
        data.xfrc_applied[self.body_id, 3:] = rotation @ torque_body

    def _executed_action(self, clipped: np.ndarray) -> np.ndarray:
        alpha = self.sensor_profile.action_alpha(self.dt)
        if alpha >= 1.0:
            self.command_state[:] = clipped
        else:
            self.command_state += alpha * (clipped - self.command_state)
        return self.command_state.astype(np.float32)

    def _sync_state_from_data(self) -> None:
        for idx, data in enumerate(self.data):
            self.position[idx] = data.qpos[:3]
            self.quaternion[idx] = data.qpos[3:7]
            self.velocity[idx] = data.qvel[:3]
            self.body_rates[idx] = data.qvel[3:6]

    def _update_ranges(self) -> None:
        rays = body_rays_world(self.quaternion)
        for sensor_idx in range(6):
            self.ranges_m[:, sensor_idx] = self.room.raycast(self.position, rays[:, sensor_idx, :])

    def _reward(self, actions: np.ndarray) -> np.ndarray:
        pos_error = np.linalg.norm(self.target_position - self.position, axis=1)
        speed = np.linalg.norm(self.velocity, axis=1)
        yaw_error = np.abs(wrap_angle(self.target_yaw - quat_to_yaw(self.quaternion)))
        clearance_penalty = np.maximum(0.0, 0.35 - np.min(self.ranges_m[:, :4], axis=1))
        control = np.linalg.norm(actions, axis=1)
        return (1.0 - pos_error - 0.15 * speed - 0.1 * yaw_error - 1.5 * clearance_penalty - 0.02 * control).astype(np.float32)


def is_mujoco_available() -> bool:
    try:
        require_mujoco()
    except ModuleNotFoundError:
        return False
    return True


def control_from_physics_profile(profile: SixDofPhysicsProfile) -> MuJoCoControlParams:
    return MuJoCoControlParams(
        mass_kg=profile.mass_kg,
        gravity=profile.gravity_m_s2,
        max_rate_rad_s=profile.max_rate_rad_s,
        thrust_scale=profile.thrust_scale,
    )
