from __future__ import annotations

import gymnasium
import numpy as np

from flightrl.mujoco.env import MuJoCoCrazyflieEnv
from flightrl.mujoco.rendering import (
    AIDECK_SOURCE_HEIGHT,
    AIDECK_SOURCE_WIDTH,
    _gap8_resize_gray4,
    require_mujoco_rendering,
)
from flightrl.mujoco.setpoint_control import (
    VisualSetpointConfig,
    firmware_setpoint_actions,
)
from flightrl.navigation.semantic_scene import SemanticScene
from flightrl.sixdof.curriculum import ResetProfile
from flightrl.sixdof.orientation import quat_to_yaw

from .contract import (
    COVERAGE_MAXIMUM_YAW_RATE_DEG_S,
    COVERAGE_OBSERVATION_DIM,
)
from .coverage import CoverageStep, CoverageTracker
from .observation import (
    build_coverage_observation,
    coverage_action_to_edge_feedback,
)


def coverage_reward(delta_coverage_score: float, *, safety_terminal: bool) -> float:
    delta = float(delta_coverage_score)
    if not np.isfinite(delta) or not 0.0 <= delta <= 1.0:
        raise ValueError("coverage score delta must be finite and in [0, 1]")
    return delta - (2.0 if safety_terminal else 0.0)


class MuJoCoCoverageEnv(gymnasium.Env[np.ndarray, np.ndarray]):
    """Single-drone simulation-only environment for a visible coverage patrol."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scene: SemanticScene,
        *,
        seed: int = 0,
        maximum_episode_steps: int = 2_500,
    ) -> None:
        if not isinstance(scene, SemanticScene):
            raise TypeError("coverage environment requires a SemanticScene")
        if type(maximum_episode_steps) is not int or maximum_episode_steps <= 0:
            raise ValueError("coverage maximum episode steps must be a positive integer")
        require_mujoco_rendering()
        self.scene = scene
        self.maximum_episode_steps = maximum_episode_steps
        self.control = VisualSetpointConfig(
            max_horizontal_speed_m_s=0.25,
            max_vertical_speed_m_s=0.0,
            max_yawrate_deg_s=COVERAGE_MAXIMUM_YAW_RATE_DEG_S,
            physics_substeps=2,
        )
        reset_profile = ResetProfile(
            name="coverage_flight_altitude",
            initial_xy_abs=0.8,
            target_xy_abs=0.8,
            z_range=(scene.flight_altitude_m, scene.flight_altitude_m),
            target_z_range=(scene.flight_altitude_m, scene.flight_altitude_m),
            attitude_std=0.0,
            target_xy_offset_abs=0.0,
            target_z_offset_abs=0.0,
            target_yaw_offset_abs=0.0,
        )
        self.sim = MuJoCoCrazyflieEnv(
            num_envs=1,
            seed=seed,
            semantic_scene=scene,
            task="position_yaw",
            reset_profile=reset_profile,
            physics_profile="crazyflie_brushless",
            max_steps=maximum_episode_steps * self.control.physics_substeps,
        )
        self.renderer = self.sim.mujoco.Renderer(
            self.sim.model,
            height=AIDECK_SOURCE_HEIGHT,
            width=AIDECK_SOURCE_WIDTH,
        )
        self.tracker = CoverageTracker(scene)
        self.observation_space = gymnasium.spaces.Box(
            -1.0,
            1.0,
            shape=(COVERAGE_OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=np.asarray((-1.0, 0.0, 0.0, -1.0), dtype=np.float32),
            high=np.asarray((1.0, 0.0, 0.0, 1.0), dtype=np.float32),
            dtype=np.float32,
        )
        self.mission_origin_position = np.zeros((1, 3), dtype=np.float32)
        self.mission_origin_yaw = np.zeros(1, dtype=np.float32)
        self.previous_edge_action = np.zeros((1, 4), dtype=np.float32)
        self.step_count = 0
        self._last_coverage_score = 0.0
        self._needs_reset = True
        self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        if options:
            raise ValueError("coverage reset options are not supported")
        self.sim.reset(seed)
        self.mission_origin_position[:] = self.sim.position
        self.mission_origin_yaw[:] = quat_to_yaw(self.sim.quaternion)
        self.previous_edge_action.fill(0.0)
        self.step_count = 0
        self.tracker.reset()
        initial_coverage = self.tracker.update(
            self.sim.position[0, :2],
            yaw_rad=float(self.mission_origin_yaw[0]),
        )
        self._last_coverage_score = initial_coverage.coverage_score
        self._needs_reset = False
        return self._observation(), self._info(
            collision=False,
            boundary_violation=False,
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self._needs_reset:
            raise RuntimeError("coverage episode is done; call reset before stepping")
        command = np.asarray(action, dtype=np.float32)
        if command.shape != (4,) or not np.isfinite(command).all():
            raise ValueError("coverage action must contain four finite values")
        if np.any(np.abs(command) > 1.0):
            raise ValueError("coverage action must stay inside normalized bounds")
        if command[1] != 0.0 or command[2] != 0.0:
            raise ValueError("coverage vy and vz actions must be structurally zero")

        terminal = False
        truncated = False
        for _ in range(self.control.physics_substeps):
            low_level = firmware_setpoint_actions(
                self.sim,
                command[None, :],
                self.control,
            )
            _obs, _reward, inner_terminal, inner_truncated, _info = self.sim.step(
                low_level
            )
            terminal = terminal or bool(inner_terminal[0])
            truncated = truncated or bool(inner_truncated[0])
            if terminal or truncated:
                break

        self.step_count += 1
        self.previous_edge_action[0] = coverage_action_to_edge_feedback(command)
        coverage = self._update_coverage(terminal)
        clearance = float(np.min(self.sim.ranges_m[0, :4]))
        delta_coverage = max(
            0.0,
            coverage.coverage_score - self._last_coverage_score,
        )
        self._last_coverage_score = coverage.coverage_score
        reward = coverage_reward(delta_coverage, safety_terminal=terminal)
        truncated = truncated or self.step_count >= self.maximum_episode_steps
        self._needs_reset = terminal or truncated
        collision = bool(self.sim.forbidden_contact_counts[0] > 0)
        boundary_violation = not bool(self.sim.room.contains(self.sim.position)[0])
        info = self._info(
            collision=collision,
            boundary_violation=boundary_violation,
        )
        info.update(
            {
                "new_visited_cells": coverage.new_visited_cells,
                "new_visible_free_cells": coverage.new_visible_free_cells,
                "position_in_free_coverage_cell": coverage.position_in_free_cell,
                "minimum_horizontal_clearance_m": clearance,
            }
        )
        return self._observation(), float(reward), terminal, truncated, info

    def close(self) -> None:
        self.renderer.close()

    def _render_frame(self) -> np.ndarray:
        self.renderer.update_scene(self.sim.data[0], camera="aideck")
        rgb = self.renderer.render().astype(np.float32)
        gray = np.clip(
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2],
            0.0,
            255.0,
        ).astype(np.uint8)
        return _gap8_resize_gray4(gray, output_width=64, output_height=48)

    def _observation(self) -> np.ndarray:
        return build_coverage_observation(
            self._render_frame()[None, ...],
            position=self.sim.position,
            velocity=self.sim.velocity,
            quaternion=self.sim.quaternion,
            body_rates=self.sim.body_rates,
            takeoff_origin_z=self.scene.room.minimum[2],
            mission_origin_position=self.mission_origin_position,
            mission_origin_yaw=self.mission_origin_yaw,
            previous_edge_action=self.previous_edge_action,
        )[0]

    def _update_coverage(self, terminal: bool) -> CoverageStep:
        if not terminal:
            return self.tracker.update(
                self.sim.position[0, :2],
                yaw_rad=float(quat_to_yaw(self.sim.quaternion)[0]),
            )
        report = self.tracker.report()
        return CoverageStep(
            new_visited_cells=0,
            new_visible_free_cells=0,
            position_in_free_cell=False,
            visited_fraction=float(report["visited_fraction"]),
            visible_free_fraction=float(report["visible_free_fraction"]),
            coverage_score=float(report["coverage_score"]),
        )

    def _info(
        self,
        *,
        collision: bool,
        boundary_violation: bool,
    ) -> dict[str, object]:
        report = self.tracker.report()
        return {
            "coverage_score": report["coverage_score"],
            "visited_fraction": report["visited_fraction"],
            "visible_free_fraction": report["visible_free_fraction"],
            "collision": collision,
            "boundary_violation": boundary_violation,
            "maximum_yaw_rate_deg_s": COVERAGE_MAXIMUM_YAW_RATE_DEG_S,
            "actor_observation_contains_range": False,
            "actor_observation_contains_map": False,
            "authority": "simulation_only",
            "flight_authority": False,
        }
