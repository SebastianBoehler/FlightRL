from __future__ import annotations

import numpy as np

from flightrl.mujoco.model import load_crazyflie_model, require_mujoco
from flightrl.mujoco.semantic_reset import (
    BOX_ROOM_RESET_CLEARANCE_M,
    SEMANTIC_RESET_CLEARANCE_M,
    sample_collision_free_reset,
)
from flightrl.mujoco.semantic_scene import box_room_from_semantic_scene
from flightrl.navigation.semantic_scene import SemanticScene
from flightrl.sixdof.curriculum import (
    ResetProfile,
    resolve_reset_profile,
    sample_initial_velocity,
    sample_reset,
)
from flightrl.sixdof.env import (
    ACTION_DIM,
    OBSERVATION_DIM,
    TASK_IDS,
    euler_to_quat,
)
from flightrl.sixdof.geometry import BoxRoom, body_rays_world
from flightrl.sixdof.physics import SixDofPhysicsProfile, resolve_physics_profile
from flightrl.sixdof.sensor_model import SixDofSensorProfile, resolve_sensor_profile
from flightrl.sixdof.validation import (
    action_batch,
    require_choice,
    require_finite_real,
    require_positive_int,
    reset_mask,
)

from .contacts import forbidden_contact_count
from .control import (
    MuJoCoControlParams,
    apply_control,
    resolve_control,
)
from .rendering import (
    render_aideck_gray as _render_aideck_gray,
    render_aideck_gray4 as _render_aideck_gray4,
    render_rgb as _render_rgb,
)
from .task_contract import apply_task_context, build_observation, reward_for_env, target_yaw_for_env


class MuJoCoCrazyflieEnv:
    """Correctness-first MuJoCo backend for the shared 6-DoF contracts."""

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
        max_steps: int = 800,
    ) -> None:
        self.num_envs = require_positive_int(num_envs, "num_envs")
        self.dt = require_finite_real(
            dt,
            "dt",
            minimum=0.0,
            strictly_greater=True,
        )
        self.task = require_choice(task, "6-DoF task", TASK_IDS)
        self.max_steps = require_positive_int(max_steps, "max_steps")
        self.mujoco = require_mujoco()
        self.reset_profile = resolve_reset_profile(reset_profile)
        self.sensor_profile = resolve_sensor_profile(sensor_profile)
        self.physics_profile = resolve_physics_profile(physics_profile)
        if room is not None and semantic_scene is not None:
            raise ValueError("provide either room or semantic_scene, not both")
        if room is not None and not isinstance(room, BoxRoom):
            raise TypeError("room must be a BoxRoom")
        if semantic_scene is not None and not isinstance(semantic_scene, SemanticScene):
            raise TypeError("semantic_scene must be a SemanticScene")
        self.semantic_scene = semantic_scene
        self.room = (
            box_room_from_semantic_scene(semantic_scene)
            if semantic_scene is not None
            else room or BoxRoom()
        )
        self.rng = np.random.default_rng(seed)
        self.control = resolve_control(control, self.physics_profile)
        self.gravity = float(self.control.gravity)
        self.max_rate = np.asarray(self.control.max_rate_rad_s, dtype=np.float32)

        self.model = load_crazyflie_model(
            self.dt,
            scene=semantic_scene,
            room=None if semantic_scene is not None else self.room,
            physics_profile=self.physics_profile,
        )
        self.body_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, "crazyflie"
        )
        if self.body_id < 0:
            raise RuntimeError("MuJoCo Crazyflie model is missing body 'crazyflie'")
        self.data = [self.mujoco.MjData(self.model) for _ in range(self.num_envs)]

        self.position = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.velocity = np.zeros_like(self.position)
        self.quaternion = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.body_rates = np.zeros_like(self.position)
        self.thrust_state = np.ones(self.num_envs, dtype=np.float64)
        self.rate_command_state = np.zeros_like(self.position, dtype=np.float64)
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
        self.forbidden_contact_counts = np.zeros(self.num_envs, dtype=np.int32)
        self.native_task_ids = np.full(
            self.num_envs,
            TASK_IDS[self.task],
            dtype=np.int32,
        )
        self.native_reward_mode_id = 0
        self.native_previous_error = np.zeros(self.num_envs, dtype=np.float32)
        self.reset(seed=seed)

    def reset(
        self, seed: int | None = None
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_mask(np.ones(self.num_envs, dtype=bool))
        self._sync_state_from_data()
        self._update_ranges()
        self.observations[:] = self.observation()
        return self.observations, []

    def reset_done(self, done: np.ndarray) -> np.ndarray:
        mask = reset_mask(done, self.num_envs)
        if np.any(mask):
            self._reset_mask(mask)
            self._sync_state_from_data()
            self._update_ranges()
            self.observations[:] = self.observation()
        return self.observations

    def step(self, actions: np.ndarray):
        clipped = action_batch(actions, self.num_envs, ACTION_DIM)
        executed = self._executed_action(clipped)
        for idx, data in enumerate(self.data):
            before = forbidden_contact_count(
                self.model,
                data,
                vehicle_body_id=self.body_id,
            )
            apply_control(self, idx, data, executed[idx])
            self.mujoco.mj_step(self.model, data)
            # Refresh final-step kinematics and contacts without a second
            # dynamics solve.
            self.mujoco.mj_fwdPosition(self.model, data)
            after = forbidden_contact_count(
                self.model,
                data,
                vehicle_body_id=self.body_id,
            )
            self.forbidden_contact_counts[idx] = max(before, after)
        self._sync_state_from_data()
        self._update_ranges()
        self.step_count += 1
        self.rewards[:] = self._reward(executed)
        outside_room = ~self.room.contains(self.position)
        forbidden_contact = self.forbidden_contact_counts > 0
        self.terminals[:] = (outside_room | forbidden_contact).astype(np.uint8)
        self.truncations[:] = (self.step_count >= self.max_steps).astype(np.uint8)
        self.previous_action[:] = executed
        self.observations[:] = self.observation()
        return self.observations, self.rewards, self.terminals, self.truncations, []

    def observation(self) -> np.ndarray:
        return build_observation(self)

    def observation_target_yaw(self) -> np.ndarray:
        return target_yaw_for_env(self)

    def set_native_context(self, **context) -> None:
        apply_task_context(self, **context)

    def render_rgb(
        self,
        width: int = 640,
        height: int = 480,
        env_index: int = 0,
        *,
        camera: str | None = None,
    ) -> np.ndarray:
        return _render_rgb(self, width, height, env_index, camera)

    def render_aideck_gray4(
        self, width: int = 64, height: int = 48, env_index: int = 0
    ) -> np.ndarray:
        return _render_aideck_gray4(self, width, height, env_index)

    def render_aideck_gray(
        self, width: int = 64, height: int = 48, env_index: int = 0
    ) -> np.ndarray:
        return _render_aideck_gray(self, width, height, env_index)

    def _reset_mask(self, mask: np.ndarray) -> None:
        count = int(np.sum(mask))
        if count == 0:
            return
        if self.semantic_scene is not None or self.room.obstacles:
            reset = sample_collision_free_reset(
                self.reset_profile,
                self.rng,
                count,
                self.room,
                clearance_m=(
                    SEMANTIC_RESET_CLEARANCE_M
                    if self.semantic_scene is not None
                    else BOX_ROOM_RESET_CLEARANCE_M
                ),
            )
        else:
            reset = sample_reset(self.reset_profile, self.rng, count, self.room)
        positions, roll, pitch, yaw, targets, target_yaw = reset
        quaternions = euler_to_quat(roll, pitch, yaw)
        velocities = sample_initial_velocity(self.reset_profile, self.rng, count)
        row = 0
        for idx, reset in enumerate(mask):
            if not reset:
                continue
            data = self.data[idx]
            self.mujoco.mj_resetData(self.model, data)
            data.qpos[:3] = positions[row]
            data.qpos[3:7] = quaternions[row]
            data.qvel[:] = 0.0
            data.qvel[:3] = velocities[row]
            self.mujoco.mj_forward(self.model, data)
            self.target_position[idx] = targets[row]
            self.target_yaw[idx] = target_yaw[row]
            row += 1
        self.command_state[mask] = 0.0
        self.thrust_state[mask] = 1.0
        self.rate_command_state[mask] = 0.0
        self.previous_action[mask] = 0.0
        self.step_count[mask] = 0
        self.rewards[mask] = 0.0
        self.terminals[mask] = 0
        self.truncations[mask] = 0
        self.forbidden_contact_counts[mask] = 0

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
        positions = np.repeat(self.position, rays.shape[1], axis=0)
        self.ranges_m[:] = self.room.raycast(
            positions,
            rays.reshape(-1, 3),
        ).reshape(self.num_envs, rays.shape[1])

    def _reward(self, actions: np.ndarray) -> np.ndarray:
        return reward_for_env(self, actions)
