from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flightrl.mujoco.semantic_observation import (
    GROUNDING_CONFIDENCE_INDEX,
    GROUNDING_HORIZONTAL_ERROR_INDEX,
)
from flightrl.mujoco.semantic_safety_replay import action_corridor_clearance
from flightrl.mujoco.semantic_exploration import (
    ActiveExplorationTeacher,
    line_of_sight_clear,
)
from flightrl.mujoco.semantic_vision_env import (
    MuJoCoSemanticVisionEnv,
    SemanticVisionEnvConfig,
)
from flightrl.mujoco.setpoint_control import VisualSetpointConfig
from flightrl.navigation.room_generation import generate_semantic_room
from flightrl.navigation.room_generation import SemanticRoomGenerationConfig
from flightrl.navigation.spatial_memory import SpatialMemoryConfig
from flightrl.sixdof.env import quat_to_yaw, wrap_angle
from flightrl.vision import VisionObservationConfig


@dataclass(frozen=True, slots=True)
class SimulatedSemanticDetectorConfig:
    horizontal_fov_deg: float = 82.0
    vertical_fov_deg: float = 62.0
    max_distance_m: float = 4.0
    dropout_probability: float = 0.08
    bearing_noise_std_rad: float = 0.02
    distance_noise_fraction: float = 0.04
    search_yawrate_deg_s: float = 20.0
    track_yawrate_deg_s: float = 8.0
    center_before_translation_deg: float = 15.0


class SemanticTrainingEnv:
    """Training hooks around the deployment-compatible semantic observation env."""

    def __init__(
        self,
        *,
        room_seed: int,
        num_envs: int,
        seed: int,
        detector: SimulatedSemanticDetectorConfig | None = None,
        active_exploration: bool = False,
        vision_config: VisionObservationConfig | None = None,
        room_profile: str = "standard",
    ) -> None:
        self.detector = detector or SimulatedSemanticDetectorConfig()
        self.active_exploration = bool(active_exploration)
        self._reset_detector_rng(seed)
        room_config = (
            SemanticRoomGenerationConfig.for_profile(room_profile)
            if self.active_exploration
            else None
        )
        env_config = (
            SemanticVisionEnvConfig(
                exploration_reward_scale=0.03,
                progress_after_target_only=True,
                success_requires_target_evidence=True,
                episode_max_steps=6_000,
            )
            if self.active_exploration
            else None
        )
        self.backend = MuJoCoSemanticVisionEnv(
            num_envs=num_envs,
            seed=seed,
            scene=generate_semantic_room(room_seed, room_config),
            vision_config=vision_config,
            memory_config=SpatialMemoryConfig(cell_size_m=0.5, local_size=16),
            control=VisualSetpointConfig(
                max_horizontal_speed_m_s=0.15 if self.active_exploration else 0.10,
                success_radius_m=0.22 if self.active_exploration else 0.16,
            ),
            config=env_config,
            action_mode=(
                "active_exploration" if self.active_exploration else "target_gated"
            ),
            auto_reset=True,
        )
        self.exploration_teacher = ActiveExplorationTeacher(
            num_envs,
            seed=seed,
        )
        self.target_observed = np.zeros(num_envs, dtype=bool)
        self.target_visible = np.zeros(num_envs, dtype=bool)
        self.target_bearing_rad = np.zeros(num_envs, dtype=np.float32)
        self.target_confidence = np.zeros(num_envs, dtype=np.float32)
        self.target_horizontal_error = np.zeros(num_envs, dtype=np.float32)
        self.reset(seed)

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._reset_detector_rng(seed)
        observations, infos = self.backend.reset(seed, write_observations=False)
        self.target_observed.fill(False)
        self.target_visible.fill(False)
        self.target_bearing_rad.fill(0.0)
        self.target_confidence.fill(0.0)
        self.target_horizontal_error.fill(0.0)
        self.exploration_teacher.reset(np.ones(self.num_agents, dtype=bool))
        self._hold_reset_altitude(np.ones(self.num_agents, dtype=bool))
        self._detect_targets(np.ones(self.num_agents, dtype=bool))
        self.backend._write_observations()
        self._write_grounding_observations()
        return observations, infos

    def step(self, actions, *, write_observations: bool = True):
        observations, rewards, terminals, truncations, infos = self.backend.step(
            actions,
            write_observations=False,
        )
        done = np.asarray(terminals) | np.asarray(truncations)
        if np.any(done):
            self.target_observed[done] = False
            self.target_visible[done] = False
            self.target_bearing_rad[done] = 0.0
            self.target_confidence[done] = 0.0
            self.target_horizontal_error[done] = 0.0
            self.exploration_teacher.reset(done)
            self._hold_reset_altitude(done)
        self._detect_targets(np.ones(self.num_agents, dtype=bool))
        if write_observations:
            self.backend._write_observations()
            self._write_grounding_observations()
        return observations, rewards, terminals, truncations, infos

    def expert_actions(self) -> np.ndarray:
        if self.active_exploration:
            return self.exploration_teacher.actions(
                self.backend,
                target_observed=self.target_observed,
                target_visible=self.target_visible,
                target_bearing_rad=self.target_bearing_rad,
            )
        positions = self.sim.position
        yaw = quat_to_yaw(self.sim.quaternion)
        target_error = self.sim.target_position - positions
        cosine, sine = np.cos(yaw), np.sin(yaw)
        body_forward = cosine * target_error[:, 0] + sine * target_error[:, 1]
        body_left = -sine * target_error[:, 0] + cosine * target_error[:, 1]
        horizontal_distance = np.linalg.norm(target_error[:, :2], axis=1)
        travel_yaw = np.arctan2(target_error[:, 1], target_error[:, 0])
        desired_yaw = np.where(
            horizontal_distance > 0.35,
            travel_yaw,
            self.sim.target_yaw,
        )
        actions = np.zeros((self.num_agents, 4), dtype=np.float32)
        actions[:, 0] = np.clip(1.8 * body_forward, -1.0, 1.0)
        actions[:, 1] = np.clip(1.8 * body_left, -1.0, 1.0)
        actions[:, 2] = np.clip(4.0 * target_error[:, 2], -1.0, 1.0)
        actions[:, 3] = np.clip(
            wrap_angle(desired_yaw - yaw) / np.deg2rad(35.0),
            -self.detector.search_yawrate_deg_s / self.control.max_yawrate_deg_s,
            self.detector.search_yawrate_deg_s / self.control.max_yawrate_deg_s,
        )
        centering_limit = np.deg2rad(self.detector.center_before_translation_deg)
        visible = self.target_visible
        actions[visible, 3] = (
            np.clip(
                self.target_bearing_rad[visible] / centering_limit,
                -1.0,
                1.0,
            )
            * self.detector.track_yawrate_deg_s
            / self.control.max_yawrate_deg_s
        )
        centering = visible & (np.abs(self.target_bearing_rad) > centering_limit)
        actions[centering, :3] = 0.0
        searching = ~self.target_observed
        actions[searching, :3] = 0.0
        actions[searching, 3] = (
            self.detector.search_yawrate_deg_s / self.control.max_yawrate_deg_s
        )
        return actions

    def close(self) -> None:
        self.backend.close()

    def _hold_reset_altitude(self, mask: np.ndarray) -> None:
        for index in np.flatnonzero(mask):
            self.sim.target_position[index, 2] = self.sim.position[index, 2]

    def _detect_targets(self, mask: np.ndarray) -> None:
        self.target_visible[mask] = False
        self.target_bearing_rad[mask] = 0.0
        self.target_confidence[mask] = 0.0
        self.target_horizontal_error[mask] = 0.0
        yaw = quat_to_yaw(self.sim.quaternion)
        horizontal_limit = np.deg2rad(self.detector.horizontal_fov_deg / 2.0)
        vertical_limit = np.deg2rad(self.detector.vertical_fov_deg / 2.0)
        for index in np.flatnonzero(mask):
            category = self.target_category(index)
            _, semantic_object = self.scene.object_by_name(category)
            center = np.asarray(semantic_object.bounds.center, dtype=np.float32)
            visible_vector = center - self.sim.position[index]
            horizontal_distance = float(np.linalg.norm(visible_vector[:2]))
            bearing = float(
                wrap_angle(
                    np.arctan2(visible_vector[1], visible_vector[0]) - yaw[index]
                )
            )
            vertical = float(
                np.arctan2(visible_vector[2], max(horizontal_distance, 1e-6))
            )
            if (
                horizontal_distance > self.detector.max_distance_m
                or abs(bearing) > horizontal_limit
                or abs(vertical) > vertical_limit
                or not line_of_sight_clear(
                    self.scene,
                    self.sim.position[index],
                    center,
                    ignored_object_id=semantic_object.object_id,
                )
                or self.detector_rng.random() < self.detector.dropout_probability
            ):
                continue
            waypoint = (
                self.sim.target_position[index, :2] - self.odometry.position_xy[index]
            )
            waypoint_distance = float(np.linalg.norm(waypoint))
            waypoint_bearing = float(
                wrap_angle(
                    np.arctan2(waypoint[1], waypoint[0]) - self.odometry.yaw[index]
                )
            )
            noisy_bearing = waypoint_bearing + self.detector_rng.normal(
                0.0,
                self.detector.bearing_noise_std_rad,
            )
            noisy_distance = waypoint_distance * (
                1.0
                + self.detector_rng.normal(
                    0.0,
                    self.detector.distance_noise_fraction,
                )
            )
            confidence = float(
                np.clip(0.95 - 0.25 * abs(bearing) / horizontal_limit, 0.5, 0.95)
            )
            self.backend.record_target_observation(
                index,
                bearing_rad=float(noisy_bearing),
                distance_m=max(0.05, float(noisy_distance)),
                confidence=confidence,
            )
            self.target_observed[index] = True
            self.target_visible[index] = True
            self.target_bearing_rad[index] = bearing
            self.target_confidence[index] = confidence
            self.target_horizontal_error[index] = np.clip(
                -bearing / horizontal_limit,
                -1.0,
                1.0,
            )

    def _reset_detector_rng(self, seed: int) -> None:
        detector_seed = np.random.SeedSequence((seed, 0xD37EC7))
        self.detector_rng = np.random.default_rng(detector_seed)

    def _write_grounding_observations(self) -> None:
        state = self.observations[:, self.layout.proprioception_slice]
        state[:, GROUNDING_CONFIDENCE_INDEX] = self.target_confidence
        state[:, GROUNDING_HORIZONTAL_ERROR_INDEX] = self.target_horizontal_error

    def target_category(self, index: int) -> str:
        from flightrl.navigation.room_generation import SEMANTIC_TARGET_CATEGORIES

        return SEMANTIC_TARGET_CATEGORIES[int(self.target_category_indices[index])]

    def front_clearance(self) -> np.ndarray:
        return self.sim.ranges_m[:, 0].copy()

    def action_corridor_clearance(self) -> np.ndarray:
        return action_corridor_clearance(self.sim.ranges_m)

    def horizontal_clearance(self) -> np.ndarray:
        return np.min(self.sim.ranges_m[:, :4], axis=1).copy()

    def navigation_clearance(self) -> np.ndarray:
        return np.min(self.sim.ranges_m[:, (0, 2, 3)], axis=1).copy()
