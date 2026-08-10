from __future__ import annotations

from collections import deque
from math import cos, pi, sin

import gymnasium
import numpy as np

from .range_contract import RANGE_ACTION_DIM, RANGE_EXPLORATION_OBSERVATION_DIM
from .range_mapper import RangeOccupancyMap, RangePose
from .range_observation import build_range_exploration_observation
from .range_safety import RangeClearanceHold, shield_range_exploration_action
from .range_world import RangeWorld


_DT_S = 0.05
_MAX_SPEED_M_S = 0.50
_MAX_YAW_RATE_RAD_S = pi / 2.0


class RangeExplorationEnv(gymnasium.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: int = 0,
        maximum_episode_steps: int = 1_200,
        stress: bool = True,
        world: RangeWorld | None = None,
        initial_pose: RangePose | None = None,
    ) -> None:
        if type(maximum_episode_steps) is not int or maximum_episode_steps <= 0:
            raise ValueError("maximum episode steps must be a positive integer")
        if type(stress) is not bool:
            raise ValueError("range exploration stress flag must be bool")
        self.maximum_episode_steps = maximum_episode_steps
        self.stress = stress
        self._fixed_world = world
        self._fixed_initial_pose = initial_pose
        self.observation_space = gymnasium.spaces.Box(
            -1.0,
            1.0,
            shape=(RANGE_EXPLORATION_OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=np.asarray((0.0, -1.0), dtype=np.float32),
            high=np.asarray((1.0, 1.0), dtype=np.float32),
            dtype=np.float32,
        )
        self.mapper = RangeOccupancyMap()
        self.world = world or RangeWorld.generate(seed)
        self.truth_pose = initial_pose or RangePose(0.0, 0.0, 0.0)
        self.estimated_pose = self.truth_pose
        self.previous_action = np.zeros(RANGE_ACTION_DIM, dtype=np.float32)
        self.step_count = 0
        self.positive_reward_total = 0.0
        self._visited_truth: set[tuple[int, int]] = set()
        self._observed_truth: set[tuple[int, int]] = set()
        self._range_bias = np.zeros(4, dtype=np.float32)
        self._odom_scale = 1.0
        self._yaw_drift_rad_s = 0.0
        self._lag_steps = 0
        self._action_queue: deque[np.ndarray] = deque()
        self._dropout_remaining = np.zeros(4, dtype=np.int8)
        self._last_ranges = np.full(4, 4.0, dtype=np.float32)
        self._last_validity = np.zeros(4, dtype=np.float32)
        self._last_safety_terminal = False
        self._last_forward_override = False
        self._clearance_hold = RangeClearanceHold()
        self._last_observation = np.zeros(
            RANGE_EXPLORATION_OBSERVATION_DIM, dtype=np.float32
        )
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
            raise ValueError("range exploration reset options are not supported")
        world_seed = (
            int(seed)
            if seed is not None
            else int(self.np_random.integers(0, np.iinfo(np.int32).max))
        )
        self.world = self._fixed_world or RangeWorld.generate(world_seed)
        self.truth_pose = self._fixed_initial_pose or self.world.sample_pose(self.np_random)
        self.estimated_pose = self.truth_pose
        self.mapper.reset()
        self.previous_action = np.zeros(RANGE_ACTION_DIM, dtype=np.float32)
        self.step_count = 0
        self.positive_reward_total = 0.0
        self._visited_truth = set()
        self._observed_truth = set()
        self._sample_stress()
        self._action_queue = deque(
            np.zeros(RANGE_ACTION_DIM, dtype=np.float32)
            for _ in range(self._lag_steps)
        )
        self._dropout_remaining.fill(0)
        self._last_safety_terminal = False
        self._last_forward_override = False
        self._clearance_hold.reset()
        self._update_truth_coverage()
        self._last_observation = self._observe(allow_dropout=False)
        self._needs_reset = False
        return self._last_observation.copy(), self._info(False)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self._needs_reset:
            raise RuntimeError("range exploration environment requires reset")
        requested = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(requested):
            raise ValueError("range exploration action violates normalized bounds")
        delayed = self._delayed_action(requested)
        executed, safety_terminal, forward_override = self._shield_action(delayed)
        self._last_safety_terminal = safety_terminal
        self._last_forward_override = forward_override
        previous_pose = self.truth_pose
        next_yaw = _wrap_angle(
            previous_pose.yaw_rad
            + float(executed[1]) * _MAX_YAW_RATE_RAD_S * _DT_S
        )
        distance = float(executed[0]) * _MAX_SPEED_M_S * _DT_S
        candidate = RangePose(
            previous_pose.x_m + cos(next_yaw) * distance,
            previous_pose.y_m + sin(next_yaw) * distance,
            next_yaw,
        )
        collision = bool(
            not safety_terminal
            and self.world.collides(candidate.x_m, candidate.y_m)
        )
        if not collision and not safety_terminal:
            self.truth_pose = candidate
        self._advance_estimate(previous_pose, self.truth_pose)
        before_visited = len(self._visited_truth)
        before_observed = len(self._observed_truth)
        self._update_truth_coverage()
        free_count = self.world.free_cell_count
        positive = (
            0.35 * (len(self._visited_truth) - before_visited) / free_count
            + 0.65 * (len(self._observed_truth) - before_observed) / free_count
        )
        reward = -2.0 if collision or safety_terminal else float(positive)
        if not collision and not safety_terminal:
            self.positive_reward_total += float(positive)
        self.previous_action = executed.copy()
        self.step_count += 1
        terminated = bool(collision or safety_terminal)
        truncated = bool(
            not terminated and self.step_count >= self.maximum_episode_steps
        )
        self._last_observation = self._observe(allow_dropout=True)
        self._needs_reset = terminated or truncated
        return (
            self._last_observation.copy(),
            reward,
            terminated,
            truncated,
            self._info(collision),
        )

    def _sample_stress(self) -> None:
        if self.stress:
            self._range_bias = self.np_random.uniform(-0.05, 0.05, 4).astype(np.float32)
            self._odom_scale = float(self.np_random.uniform(0.85, 1.15))
            self._yaw_drift_rad_s = float(self.np_random.uniform(-2.0, 2.0) * pi / 180.0)
            self._lag_steps = int(self.np_random.choice((0, 2, 5)))
        else:
            self._range_bias = np.zeros(4, dtype=np.float32)
            self._odom_scale = 1.0
            self._yaw_drift_rad_s = 0.0
            self._lag_steps = 0

    def _delayed_action(self, requested: np.ndarray) -> np.ndarray:
        if self._lag_steps == 0:
            return requested.copy()
        self._action_queue.append(requested.copy())
        return self._action_queue.popleft()

    def _advance_estimate(self, before: RangePose, after: RangePose) -> None:
        self.estimated_pose = RangePose(
            self.estimated_pose.x_m + (after.x_m - before.x_m) * self._odom_scale,
            self.estimated_pose.y_m + (after.y_m - before.y_m) * self._odom_scale,
            _wrap_angle(
                self.estimated_pose.yaw_rad
                + _wrap_angle(after.yaw_rad - before.yaw_rad)
                + self._yaw_drift_rad_s * _DT_S
            ),
        )

    def _observe(self, *, allow_dropout: bool) -> np.ndarray:
        ranges, validity = self.world.horizontal_ranges(self.truth_pose)
        finite = validity.astype(bool)
        ranges[finite] = np.clip(
            ranges[finite] + self._range_bias[finite], 0.03, 4.0
        )
        if allow_dropout and self.stress:
            self._dropout_remaining = np.maximum(self._dropout_remaining - 1, 0)
            if float(self.np_random.random()) < 0.02:
                sensor = int(self.np_random.integers(0, 4))
                self._dropout_remaining[sensor] = int(self.np_random.choice((1, 3)))
            dropped = self._dropout_remaining > 0
            ranges[dropped] = 4.0
            validity[dropped] = 0.0
        self.mapper.update(self.estimated_pose, ranges, validity)
        self._last_ranges = ranges.copy()
        self._last_validity = validity.copy()
        return build_range_exploration_observation(
            self.mapper.exploration_crop(self.estimated_pose),
            ranges / 4.0,
            validity,
            self.previous_action,
        )

    def _update_truth_coverage(self) -> None:
        cell = self.world.truth_cell(self.truth_pose.x_m, self.truth_pose.y_m)
        if cell is not None and not self.world.occupied[cell]:
            self._visited_truth.add(cell)
        self._observed_truth.update(self.world.visible_free_cells(self.truth_pose))

    def _info(self, collision: bool) -> dict[str, object]:
        return {
            "collision": bool(collision),
            "safety_terminal": self._last_safety_terminal,
            "forward_clearance_override": self._last_forward_override,
            "coverage_fraction": len(self._observed_truth) / self.world.free_cell_count,
            "visited_fraction": len(self._visited_truth) / self.world.free_cell_count,
            "positive_reward_total": self.positive_reward_total,
            "frontier_count": len(self.mapper.frontier_cells(self.estimated_pose)),
            "truth_exposed_to_actor": False,
        }

    def _shield_action(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        result, safety_terminal, reasons = shield_range_exploration_action(
            action,
            self._last_ranges,
            self._last_validity,
            self._last_observation[:4096].reshape(4, 32, 32),
        )
        result, reasons = self._clearance_hold.apply(result, reasons)
        forward_override = bool(
            safety_terminal
            or "forward_clearance_override" in reasons
            or "horizontal_clearance_override" in reasons
            or "estimated_map_clearance_override" in reasons
            or "clearance_hold" in reasons
        )
        return result, safety_terminal, forward_override


def _wrap_angle(value: float) -> float:
    return (value + pi) % (2.0 * pi) - pi
