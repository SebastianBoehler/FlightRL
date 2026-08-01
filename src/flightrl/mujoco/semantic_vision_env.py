from __future__ import annotations

import gymnasium
import numpy as np

from flightrl.mujoco.camera_model import sample_gray4_camera_parameters
from flightrl.mujoco.env import MuJoCoCrazyflieEnv
from flightrl.mujoco.odometry import SimulatedOdometry
from flightrl.mujoco.semantic_observation import (
    SemanticStudentObservationLayout,
    write_semantic_observations,
)
from flightrl.mujoco.setpoint_control import (
    VisualSetpointConfig,
    firmware_setpoint_actions,
)
from flightrl.mujoco.semantic_task import (
    SemanticVisionEnvConfig,
    project_semantic_actions,
    semantic_episode_info,
    semantic_rewards,
)
from flightrl.navigation.mission_spec import TargetAnchor
from flightrl.navigation.room_generation import (
    SEMANTIC_TARGET_CATEGORIES,
    SemanticRoomGenerationConfig,
    generate_semantic_room,
)
from flightrl.navigation.semantic_scene import SemanticScene
from flightrl.navigation.spatial_memory import (
    EgocentricSpatialMemory,
    SpatialMemoryConfig,
)
from flightrl.vision import VisionObservationConfig, VisionObservationEncoder


class MuJoCoSemanticVisionEnv:
    """Target-conditioned visual navigation without target-coordinate observations."""

    def __init__(
        self,
        num_envs: int = 4,
        seed: int = 0,
        *,
        scene: SemanticScene | None = None,
        room_config: SemanticRoomGenerationConfig | None = None,
        memory_config: SpatialMemoryConfig | None = None,
        vision_config: VisionObservationConfig | None = None,
        control: VisualSetpointConfig | None = None,
        config: SemanticVisionEnvConfig | None = None,
        action_mode: str = "target_gated",
        auto_reset: bool = True,
    ) -> None:
        self.num_agents = int(num_envs)
        self._reset_random_streams(seed)
        self.scene = scene or generate_semantic_room(seed, room_config)
        self.control = control or VisualSetpointConfig()
        self.config = config or SemanticVisionEnvConfig()
        self.semantic_action_mode = action_mode
        if action_mode not in {"target_gated", "active_exploration"}:
            raise ValueError(f"unknown semantic action mode {action_mode!r}")
        self.auto_reset = bool(auto_reset)
        self.vision_config = vision_config or VisionObservationConfig(
            width=64,
            height=48,
            color_mode="grayscale",
            frame_stack=1,
            include_delta=True,
            include_motion_mask=True,
            normalization="minus_one_one",
        )
        self.memory_config = memory_config or SpatialMemoryConfig()
        self.layout = SemanticStudentObservationLayout(
            self.vision_config,
            self.memory_config,
        )
        self.single_observation_space = gymnasium.spaces.Box(
            -1.0,
            1.0,
            shape=(self.layout.flat_dim,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Box(
            -1.0, 1.0, shape=(4,), dtype=np.float32
        )
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
        self.sim = MuJoCoCrazyflieEnv(
            num_envs=self.num_agents,
            seed=seed,
            task="position_yaw",
            semantic_scene=self.scene,
            max_steps=self.config.episode_max_steps,
        )
        self.renderer = self.sim.mujoco.Renderer(
            self.sim.model,
            height=self.vision_config.height,
            width=self.vision_config.width,
        )
        self.encoders = tuple(
            VisionObservationEncoder(self.vision_config) for _ in range(self.num_agents)
        )
        self.memories = tuple(
            EgocentricSpatialMemory(self.scene.room, self.memory_config)
            for _ in range(self.num_agents)
        )
        self.target_category_indices = np.zeros(self.num_agents, dtype=np.intp)
        self.target_means = np.empty(self.num_agents, dtype=np.float32)
        self.gammas = np.empty(self.num_agents, dtype=np.float32)
        self.odometry = SimulatedOdometry(self.num_agents, self.config.odometry)
        self.previous_distance = np.zeros(self.num_agents, dtype=np.float32)
        self.target_acquired = np.zeros(self.num_agents, dtype=bool)
        self.episode_return = np.zeros(self.num_agents, dtype=np.float32)
        self.tick = 0
        self._validate_scene_targets()
        self.reset(seed)

    def reset(self, seed=None, *, write_observations: bool = True):
        if seed is not None:
            self._reset_random_streams(seed)
        self.sim.reset(seed)
        mask = np.ones(self.num_agents, dtype=bool)
        for encoder, memory in zip(self.encoders, self.memories, strict=True):
            encoder.reset()
            memory.reset()
        self._assign_targets(mask)
        self.target_acquired.fill(False)
        self.odometry.reset(self.sim, self.odometry_rng, mask)
        self._update_memories(mask)
        self._sample_camera_randomization(mask)
        self.previous_distance[:] = self._distance_to_target()
        self.episode_return.fill(0.0)
        self.rewards.fill(0.0)
        self.terminals.fill(False)
        self.truncations.fill(False)
        self.tick = 0
        if write_observations:
            self._write_observations()
        return self.observations, []

    def step(self, actions, *, write_observations: bool = True):
        commands = project_semantic_actions(
            actions,
            action_mode=self.semantic_action_mode,
            max_yawrate_deg_s=self.control.max_yawrate_deg_s,
        )
        previous_distance = self._distance_to_target()
        collisions = np.zeros(self.num_agents, dtype=bool)
        truncations = np.zeros(self.num_agents, dtype=bool)
        for _ in range(self.control.physics_substeps):
            low_level = firmware_setpoint_actions(self.sim, commands, self.control)
            _obs, _reward, terminal, truncated, _info = self.sim.step(low_level)
            collisions |= terminal.astype(bool)
            truncations |= truncated.astype(bool)

        distance = self._distance_to_target()
        success = distance <= self.control.success_radius_m
        if self.config.success_requires_target_evidence:
            success &= self.target_acquired
        done = collisions | truncations | success
        self.odometry.advance(self.sim, self.odometry_rng)
        new_cells = self._update_memories(~done)
        self.rewards[:] = semantic_rewards(
            self.config,
            previous_distance=previous_distance,
            distance=distance,
            target_acquired=self.target_acquired,
            new_cells=new_cells,
            commands=commands,
            front_clearance=self.sim.ranges_m[:, 0],
            collisions=collisions,
            success=success,
        )
        self.episode_return += self.rewards
        self.terminals[:] = collisions | success
        self.truncations[:] = truncations
        info = semantic_episode_info(
            done,
            success,
            collisions,
            self.target_acquired,
            self.episode_return,
        )
        if np.any(done) and self.auto_reset:
            self.sim.reset_done(done)
            for index in np.flatnonzero(done):
                self.encoders[index].reset()
                self.memories[index].reset()
            self._assign_targets(done)
            self.target_acquired[done] = False
            self.odometry.reset(self.sim, self.odometry_rng, done)
            self._update_memories(done)
            self._sample_camera_randomization(done)
            self.episode_return[done] = 0.0
        self.previous_distance[:] = self._distance_to_target()
        if write_observations:
            self._write_observations()
        self.tick += 1
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def close(self) -> None:
        self.renderer.close()

    def record_target_observation(
        self,
        env_index: int,
        *,
        bearing_rad: float,
        distance_m: float,
        confidence: float,
    ) -> None:
        self.memories[env_index].observe_semantic(
            self.odometry.position_xy[env_index],
            float(self.odometry.yaw[env_index]),
            bearing_rad,
            distance_m,
            confidence,
            replace=True,
        )
        self.target_acquired[env_index] = True

    def _validate_scene_targets(self) -> None:
        for category in SEMANTIC_TARGET_CATEGORIES:
            self.scene.object_by_name(category)

    def _assign_targets(self, mask: np.ndarray) -> None:
        for index in np.flatnonzero(mask):
            category_index = int(self.rng.integers(len(SEMANTIC_TARGET_CATEGORIES)))
            category = SEMANTIC_TARGET_CATEGORIES[category_index]
            resolved = self.scene.resolve_target(
                category,
                anchor=TargetAnchor.PREFERRED,
                reference_position_m=tuple(
                    float(value) for value in self.sim.position[index]
                ),
            )
            self.target_category_indices[index] = category_index
            self.sim.target_position[index] = resolved.position_m
            self.sim.target_yaw[index] = resolved.yaw_rad

    def _update_memories(self, mask: np.ndarray) -> np.ndarray:
        new_cells = np.zeros(self.num_agents, dtype=np.float32)
        bearings = np.asarray((0.0, np.pi, np.pi / 2.0, -np.pi / 2.0))
        for index in np.flatnonzero(mask):
            memory = self.memories[index]
            position = self.odometry.position_xy[index]
            new_cells[index] = memory.update_pose(position)
            if self.config.use_range_map_updates:
                memory.observe_rays(
                    position,
                    float(self.odometry.yaw[index]),
                    bearings,
                    self.sim.ranges_m[index, :4],
                    max_range_m=self.sim.room.max_range_m,
                )
        return new_cells

    def _write_observations(self) -> None:
        write_semantic_observations(
            observations=self.observations,
            layout=self.layout,
            renderer=self.renderer,
            sim=self.sim,
            encoders=self.encoders,
            memories=self.memories,
            odometry=self.odometry,
            target_category_indices=self.target_category_indices,
            target_means=self.target_means,
            gammas=self.gammas,
            rng=self.camera_rng,
        )

    def _sample_camera_randomization(self, mask: np.ndarray) -> None:
        sample_gray4_camera_parameters(
            self.camera_rng,
            mask,
            self.target_means,
            self.gammas,
        )

    def _reset_random_streams(self, seed: int) -> None:
        task_seed, odometry_seed, camera_seed = np.random.SeedSequence(seed).spawn(3)
        self.task_rng = np.random.default_rng(task_seed)
        self.odometry_rng = np.random.default_rng(odometry_seed)
        self.camera_rng = np.random.default_rng(camera_seed)
        self.rng = self.task_rng

    def _distance_to_target(self) -> np.ndarray:
        return np.linalg.norm(
            self.sim.target_position - self.sim.position,
            axis=1,
        ).astype(np.float32)
