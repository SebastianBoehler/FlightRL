from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flightrl.mujoco.semantic_planning import PrivilegedGridPlanner
from flightrl.navigation.semantic_scene import Bounds3D, SemanticScene
from flightrl.sixdof.env import quat_to_yaw, wrap_angle


@dataclass(frozen=True, slots=True)
class ActiveExplorationTeacherConfig:
    forward_clearance_m: float = 0.65
    side_clearance_m: float = 0.45
    cruise_action: float = 0.85
    avoidance_steps: int = 90
    initial_scan_steps: int = 900
    viewpoint_scan_steps: int = 450
    periodic_scan_steps: int = 225
    cruise_steps: int = 500
    viewpoint_radius_m: float = 0.30
    max_yawrate_deg_s: float = 20.0
    center_before_forward_deg: float = 20.0


class ActiveExplorationTeacher:
    """Privileged camera-centric teacher; range and target pose stay out of observations."""

    def __init__(
        self,
        num_envs: int,
        *,
        seed: int,
        config: ActiveExplorationTeacherConfig | None = None,
    ) -> None:
        self.config = config or ActiveExplorationTeacherConfig()
        self.rng = np.random.default_rng(seed)
        self.turn_direction = np.ones(num_envs, dtype=np.float32)
        self.avoidance_remaining = np.zeros(num_envs, dtype=np.int32)
        self.scan_remaining = np.zeros(num_envs, dtype=np.int32)
        self.cruise_remaining = np.zeros(num_envs, dtype=np.int32)
        self.paths: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self.path_goals = np.full((num_envs, 2), np.nan, dtype=np.float32)
        self.replan_remaining = np.zeros(num_envs, dtype=np.int32)
        self.coverage_goal_indices = np.full(num_envs, -1, dtype=np.int32)
        self.coverage_visited = np.zeros((num_envs, 5), dtype=bool)
        self.planner: PrivilegedGridPlanner | None = None
        self.reset(np.ones(num_envs, dtype=bool))

    def reset(self, mask: np.ndarray) -> None:
        self.turn_direction[mask] = 1.0
        self.avoidance_remaining[mask] = 0
        self.scan_remaining[mask] = self.config.initial_scan_steps
        self.cruise_remaining[mask] = self.config.cruise_steps
        self.replan_remaining[mask] = 0
        self.coverage_goal_indices[mask] = -1
        self.coverage_visited[mask] = False
        self.path_goals[mask] = np.nan
        for index in np.flatnonzero(mask):
            self.paths[index] = []

    def actions(
        self,
        env,
        *,
        target_observed: np.ndarray,
        target_visible: np.ndarray,
        target_bearing_rad: np.ndarray,
    ) -> np.ndarray:
        ranges = env.sim.ranges_m
        front = ranges[:, 0]
        left = ranges[:, 2]
        right = ranges[:, 3]
        side_blocked = np.minimum(left, right) < self.config.side_clearance_m
        blocked = (front < self.config.forward_clearance_m) | side_blocked
        newly_blocked = blocked & (self.avoidance_remaining <= 0)
        choose_left = np.where(
            left < self.config.side_clearance_m,
            False,
            np.where(
                right < self.config.side_clearance_m,
                True,
                left >= right,
            ),
        )
        self.turn_direction[newly_blocked] = np.where(
            choose_left[newly_blocked],
            1.0,
            -1.0,
        )
        self.avoidance_remaining[newly_blocked] = self.config.avoidance_steps

        avoiding = self.avoidance_remaining > 0
        self.avoidance_remaining[avoiding] -= 1
        yaw = quat_to_yaw(env.sim.quaternion)
        self._prepare_coverage(env, target_observed=target_observed)
        desired_bearing = self._planned_bearing(
            env,
            yaw,
            target_observed=target_observed,
        )
        desired_bearing[target_visible] = target_bearing_rad[target_visible]
        searching = ~target_observed
        scanning = searching & (self.scan_remaining > 0)
        self.scan_remaining[scanning] -= 1
        cruising = searching & ~scanning
        self.cruise_remaining[cruising] -= 1
        restart_scan = cruising & (self.cruise_remaining <= 0)
        self.scan_remaining[restart_scan] = self.config.periodic_scan_steps
        self.cruise_remaining[restart_scan] = self.config.cruise_steps

        actions = np.zeros((env.num_agents, 4), dtype=np.float32)
        actions[:, 0] = self.config.cruise_action
        yaw_limit = np.deg2rad(self.config.max_yawrate_deg_s)
        actions[:, 3] = np.clip(
            desired_bearing / yaw_limit,
            -1.0,
            1.0,
        )
        not_centered = np.abs(desired_bearing) > np.deg2rad(
            self.config.center_before_forward_deg
        )
        actions[not_centered, 0] = 0.0
        actions[scanning, 0] = 0.0
        actions[scanning, 3] = self.turn_direction[scanning]

        actions[avoiding, 3] = self.turn_direction[avoiding]
        actions[avoiding, 0] = 0.0
        actions[:, 3] *= self.config.max_yawrate_deg_s / env.control.max_yawrate_deg_s
        return actions

    def _prepare_coverage(
        self,
        env,
        *,
        target_observed: np.ndarray,
    ) -> None:
        planner = self._planner(env)
        goals = planner.coverage_goals()
        positions = env.sim.position[:, :2]
        eligible = (
            ~target_observed
            & (self.scan_remaining <= 0)
            & (self.avoidance_remaining <= 0)
        )
        for index in np.flatnonzero(eligible):
            current = self.coverage_goal_indices[index]
            if (
                current >= 0
                and np.linalg.norm(goals[current] - positions[index])
                <= self.config.viewpoint_radius_m
            ):
                self.coverage_visited[index, current] = True
                self.coverage_goal_indices[index] = -1
                self.scan_remaining[index] = self.config.viewpoint_scan_steps
                self.paths[index] = []
                self.path_goals[index] = np.nan
                continue
            if current >= 0:
                continue
            available = np.flatnonzero(~self.coverage_visited[index])
            if len(available) == 0:
                self.coverage_visited[index] = False
                available = np.arange(len(goals))
            distances = [
                np.linalg.norm(goals[goal_index] - positions[index])
                for goal_index in available
            ]
            self.coverage_goal_indices[index] = int(
                available[int(np.argmin(distances))]
            )

    def _planned_bearing(
        self,
        env,
        yaw: np.ndarray,
        *,
        target_observed: np.ndarray,
    ) -> np.ndarray:
        planner = self._planner(env)
        positions = env.sim.position[:, :2]
        bearings = np.zeros(env.num_agents, dtype=np.float32)
        for index in np.flatnonzero(target_observed):
            goal = env.sim.target_position[index, :2]
            bearings[index] = self._path_bearing(
                planner,
                index,
                positions[index],
                goal,
                yaw[index],
            )
        for index in np.flatnonzero(
            ~target_observed & (self.coverage_goal_indices >= 0)
        ):
            goal = planner.coverage_goals()[self.coverage_goal_indices[index]]
            bearings[index] = self._path_bearing(
                planner,
                index,
                positions[index],
                goal,
                yaw[index],
            )
        return bearings

    def _path_bearing(
        self,
        planner: PrivilegedGridPlanner,
        index: int,
        position: np.ndarray,
        goal: np.ndarray,
        yaw: float,
    ) -> float:
        goal = np.asarray(goal, dtype=np.float32)
        goal_changed = not np.allclose(
            goal,
            self.path_goals[index],
            atol=0.05,
        )
        if (
            goal_changed
            or not self.paths[index]
            or self.replan_remaining[index] <= 0
        ):
            self.paths[index] = planner.path(position, goal)
            self.path_goals[index] = goal
            self.replan_remaining[index] = 50
        else:
            self.replan_remaining[index] -= 1
        while (
            len(self.paths[index]) > 1
            and np.linalg.norm(self.paths[index][0] - position) < 0.18
        ):
            self.paths[index].pop(0)
        waypoint = self.paths[index][0] if self.paths[index] else goal
        return float(
            wrap_angle(
                np.arctan2(
                    waypoint[1] - position[1],
                    waypoint[0] - position[0],
                )
                - yaw
            )
        )

    def _planner(self, env) -> PrivilegedGridPlanner:
        if self.planner is None:
            self.planner = PrivilegedGridPlanner(env.scene)
        return self.planner


def line_of_sight_clear(
    scene: SemanticScene,
    origin: np.ndarray,
    target: np.ndarray,
    *,
    ignored_object_id: str,
) -> bool:
    direction = np.asarray(target, dtype=np.float32) - np.asarray(
        origin,
        dtype=np.float32,
    )
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-6:
        return True
    for obj in scene.objects:
        if not obj.collision or obj.object_id == ignored_object_id:
            continue
        hit = _segment_box_hit_fraction(
            np.asarray(origin, dtype=np.float32),
            direction,
            obj.bounds,
        )
        if hit is not None and hit * distance < distance - 0.05:
            return False
    return True


def _segment_box_hit_fraction(
    origin: np.ndarray,
    direction: np.ndarray,
    bounds: Bounds3D,
) -> float | None:
    lower = np.asarray(bounds.minimum, dtype=np.float32)
    upper = np.asarray(bounds.maximum, dtype=np.float32)
    minimum = 0.0
    maximum = 1.0
    for axis in range(3):
        if abs(float(direction[axis])) < 1e-8:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return None
            continue
        first = float((lower[axis] - origin[axis]) / direction[axis])
        second = float((upper[axis] - origin[axis]) / direction[axis])
        near, far = min(first, second), max(first, second)
        minimum = max(minimum, near)
        maximum = min(maximum, far)
        if minimum > maximum:
            return None
    return minimum
