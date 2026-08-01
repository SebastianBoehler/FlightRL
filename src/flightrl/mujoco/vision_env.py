from __future__ import annotations

import gymnasium
import numpy as np

from flightrl.mujoco.camera_model import (
    randomize_gray4_frame,
    sample_gray4_camera_parameters,
)
from flightrl.mujoco.env import MuJoCoCrazyflieEnv
from flightrl.mujoco.rendering import require_mujoco_rendering
from flightrl.mujoco.setpoint_control import VisualSetpointConfig, firmware_setpoint_actions
from flightrl.navigation.semantic_scene import SemanticScene
from flightrl.sixdof.env import quat_to_yaw, wrap_angle
from flightrl.sixdof.geometry import quat_to_matrix
from flightrl.vision import VisionObservationConfig, VisionObservationEncoder


INTENT_DIM = 6


class MuJoCoVisionPufferEnv:
    """Correctness-first pixel environment; rendering throughput is not yet optimized."""

    def __init__(
        self,
        num_envs: int = 4,
        seed: int = 0,
        control: VisualSetpointConfig | None = None,
        semantic_scene: SemanticScene | None = None,
    ) -> None:
        require_mujoco_rendering()
        self.num_agents = int(num_envs)
        self.vision_config = VisionObservationConfig(
            width=64,
            height=48,
            color_mode="grayscale",
            frame_stack=1,
            include_delta=True,
            include_motion_mask=True,
            normalization="minus_one_one",
        )
        self.single_observation_space = gymnasium.spaces.Box(
            -1.0,
            1.0,
            shape=(self.vision_config.flat_dim + INTENT_DIM,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = gymnasium.vector.utils.batch_space(
            self.single_observation_space,
            self.num_agents,
        )
        self.action_space = gymnasium.vector.utils.batch_space(
            self.single_action_space,
            self.num_agents,
        )
        self.observations = np.empty(self.observation_space.shape, dtype=np.float32)
        self.rewards = np.zeros(self.num_agents, dtype=np.float32)
        self.terminals = np.zeros(self.num_agents, dtype=bool)
        self.truncations = np.zeros(self.num_agents, dtype=bool)
        self.control = control or VisualSetpointConfig()
        self.sim = MuJoCoCrazyflieEnv(
            num_envs=self.num_agents,
            seed=seed,
            task="position_yaw",
            semantic_scene=semantic_scene,
        )
        self.encoders = [VisionObservationEncoder(self.vision_config) for _ in range(self.num_agents)]
        self.renderer = self.sim.mujoco.Renderer(
            self.sim.model,
            height=self.vision_config.height,
            width=self.vision_config.width,
        )
        self.rng = np.random.default_rng(seed)
        self.target_means = np.empty(self.num_agents, dtype=np.float32)
        self.gammas = np.empty(self.num_agents, dtype=np.float32)
        self.previous_distance = np.zeros(self.num_agents, dtype=np.float32)
        self.tick = 0
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.sim.reset(seed)
        for encoder in self.encoders:
            encoder.reset()
        self._sample_camera_randomization(np.ones(self.num_agents, dtype=bool))
        self.previous_distance[:] = self._distance_to_target()
        self._write_observations()
        self.rewards.fill(0.0)
        self.terminals.fill(False)
        self.truncations.fill(False)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        commands = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        previous_distance = self._distance_to_target()
        terminals = np.zeros(self.num_agents, dtype=bool)
        truncations = np.zeros(self.num_agents, dtype=bool)
        for _ in range(self.control.physics_substeps):
            low_level = self._firmware_controller_actions(commands)
            _obs, _reward, terminal, truncated, _info = self.sim.step(low_level)
            terminals |= terminal.astype(bool)
            truncations |= truncated.astype(bool)

        distance = self._distance_to_target()
        success = distance <= self.control.success_radius_m
        done = terminals | truncations | success
        progress = previous_distance - distance
        self.rewards[:] = (
            8.0 * progress
            - 0.005 * np.sum(commands * commands, axis=1)
            - 5.0 * terminals
            + 5.0 * success
        )
        self.terminals[:] = terminals | success
        self.truncations[:] = truncations
        info = self._episode_info(success, terminals)
        if np.any(done):
            self.sim.reset_done(done)
            for index in np.flatnonzero(done):
                self.encoders[index].reset()
            self._sample_camera_randomization(done)
        self.previous_distance[:] = self._distance_to_target()
        self._write_observations()
        self.tick += 1
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def close(self) -> None:
        self.renderer.close()

    def _firmware_controller_actions(self, commands: np.ndarray) -> np.ndarray:
        return firmware_setpoint_actions(self.sim, commands, self.control)

    def _write_observations(self) -> None:
        intent = self._intent_observation()
        for index, encoder in enumerate(self.encoders):
            self.renderer.update_scene(self.sim.data[index], camera="aideck")
            rgb = self.renderer.render().astype(np.float32)
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            frame = self._camera_randomization(gray, index)
            self.observations[index, : self.vision_config.flat_dim] = encoder.encode_flat(frame)
            self.observations[index, self.vision_config.flat_dim :] = intent[index]

    def _intent_observation(self) -> np.ndarray:
        delta_world = self.sim.target_position - self.sim.position
        rotation = quat_to_matrix(self.sim.quaternion)
        delta_body = np.einsum("nji,nj->ni", rotation, delta_world, optimize=True)
        distance = np.linalg.norm(delta_body, axis=1, keepdims=True)
        direction = delta_body / np.maximum(distance, 1e-6)
        yaw_error = wrap_angle(self.sim.target_yaw - quat_to_yaw(self.sim.quaternion))
        return np.column_stack(
            (
                direction,
                np.clip(distance[:, 0] / 4.0, 0.0, 1.0),
                np.sin(yaw_error),
                np.cos(yaw_error),
            )
        ).astype(np.float32)

    def _camera_randomization(self, gray: np.ndarray, index: int) -> np.ndarray:
        return randomize_gray4_frame(
            gray,
            target_mean=float(self.target_means[index]),
            gamma=float(self.gammas[index]),
            rng=self.rng,
        )

    def _sample_camera_randomization(self, mask: np.ndarray) -> None:
        sample_gray4_camera_parameters(
            self.rng,
            mask,
            self.target_means,
            self.gammas,
        )

    def _distance_to_target(self) -> np.ndarray:
        return np.linalg.norm(self.sim.target_position - self.sim.position, axis=1).astype(np.float32)

    def _episode_info(self, success: np.ndarray, collision: np.ndarray) -> list[dict[str, float]]:
        if self.tick % 64:
            return []
        return [
            {
                "reward": float(np.mean(self.rewards)),
                "success_rate": float(np.mean(success)),
                "collision_rate": float(np.mean(collision)),
                "distance_m": float(np.mean(self._distance_to_target())),
            }
        ]
