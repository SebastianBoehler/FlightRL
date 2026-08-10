from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from flightrl.mujoco.camera_contract import AIDECK_HORIZONTAL_FOV_DEG
from flightrl.mujoco.semantic_exploration import line_of_sight_clear
from flightrl.mujoco.semantic_planning import PrivilegedGridConfig, PrivilegedGridPlanner
from flightrl.navigation.semantic_scene import SemanticScene


SCHEMA = "flightrl.coverage_exploration_score.v1"


@dataclass(frozen=True, slots=True)
class CoverageTrackerConfig:
    cell_size_m: float = 0.25
    horizontal_field_of_view_deg: float = AIDECK_HORIZONTAL_FOV_DEG
    maximum_visible_distance_m: float = 4.0

    def __post_init__(self) -> None:
        values = (
            self.cell_size_m,
            self.horizontal_field_of_view_deg,
            self.maximum_visible_distance_m,
        )
        if not all(isfinite(value) and value > 0.0 for value in values):
            raise ValueError("coverage tracker values must be finite and positive")
        if self.horizontal_field_of_view_deg >= 180.0:
            raise ValueError("coverage horizontal field of view must be below 180 degrees")


@dataclass(frozen=True, slots=True)
class CoverageStep:
    new_visited_cells: int
    new_visible_free_cells: int
    position_in_free_cell: bool
    visited_fraction: float
    visible_free_fraction: float
    coverage_score: float


class CoverageTracker:
    """Privileged simulator scorer; its grid is never part of actor observations."""

    def __init__(
        self,
        scene: SemanticScene,
        config: CoverageTrackerConfig | None = None,
    ) -> None:
        self.scene = scene
        self.config = config or CoverageTrackerConfig()
        self.planner = PrivilegedGridPlanner(
            scene,
            PrivilegedGridConfig(cell_size_m=self.config.cell_size_m),
        )
        self.visited = np.zeros_like(self.planner.blocked)
        self.visible = np.zeros_like(self.planner.blocked)
        self._free_indices = np.argwhere(~self.planner.blocked)
        self._free_centers = np.column_stack(
            (
                self.planner.xs[self._free_indices[:, 0]],
                self.planner.ys[self._free_indices[:, 1]],
            )
        ).astype(np.float32)

    @property
    def visited_count(self) -> int:
        return int(self.visited.sum())

    @property
    def visible_count(self) -> int:
        return int(self.visible.sum())

    @property
    def free_cell_count(self) -> int:
        return len(self._free_indices)

    def reset(self) -> None:
        self.visited.fill(False)
        self.visible.fill(False)

    def update(self, position_xy, *, yaw_rad: float) -> CoverageStep:
        position = np.asarray(position_xy, dtype=np.float32)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("coverage position must be finite XY")
        if not isfinite(yaw_rad):
            raise ValueError("coverage yaw must be finite")
        if not self.scene.room.contains(
            (float(position[0]), float(position[1]), self.scene.flight_altitude_m)
        ):
            raise ValueError("coverage position is outside the scene room")
        cell = self._free_cell(position)

        visited_before = self.visited_count
        visible_before = self.visible_count
        if cell is not None:
            self.visited[cell] = True
        self._mark_visible(position, float(yaw_rad))
        return CoverageStep(
            new_visited_cells=self.visited_count - visited_before,
            new_visible_free_cells=self.visible_count - visible_before,
            position_in_free_cell=cell is not None,
            visited_fraction=self.visited_count / self.free_cell_count,
            visible_free_fraction=self.visible_count / self.free_cell_count,
            coverage_score=self._coverage_score(),
        )

    def report(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "cell_size_m": self.config.cell_size_m,
            "horizontal_field_of_view_deg": self.config.horizontal_field_of_view_deg,
            "maximum_visible_distance_m": self.config.maximum_visible_distance_m,
            "free_cells": self.free_cell_count,
            "visited_cells": self.visited_count,
            "visible_free_cells": self.visible_count,
            "visited_fraction": self.visited_count / self.free_cell_count,
            "visible_free_fraction": self.visible_count / self.free_cell_count,
            "coverage_score": self._coverage_score(),
            "coverage_score_formula": (
                "0.5 * visited_fraction + 0.5 * visible_free_fraction"
            ),
            "privileged_simulator_scorer": True,
            "actor_observation_contains_map": False,
            "training_authority": False,
            "deployment_authority": False,
            "flight_authority": False,
        }

    def _coverage_score(self) -> float:
        return 0.5 * (self.visited_count + self.visible_count) / self.free_cell_count

    def _free_cell(self, position: np.ndarray) -> tuple[int, int] | None:
        half_cell = self.config.cell_size_m / 2.0
        if (
            position[0] < self.planner.xs[0] - half_cell
            or position[0] > self.planner.xs[-1] + half_cell
            or position[1] < self.planner.ys[0] - half_cell
            or position[1] > self.planner.ys[-1] + half_cell
        ):
            return None
        x_index = int(np.argmin(np.abs(self.planner.xs - position[0])))
        y_index = int(np.argmin(np.abs(self.planner.ys - position[1])))
        if self.planner.blocked[x_index, y_index]:
            return None
        return x_index, y_index

    def _mark_visible(self, position: np.ndarray, yaw_rad: float) -> None:
        deltas = self._free_centers - position
        distances = np.linalg.norm(deltas, axis=1)
        bearings = np.arctan2(deltas[:, 1], deltas[:, 0])
        errors = np.arctan2(np.sin(bearings - yaw_rad), np.cos(bearings - yaw_rad))
        half_fov = np.deg2rad(self.config.horizontal_field_of_view_deg / 2.0)
        candidates = (distances <= self.config.maximum_visible_distance_m) & (
            (distances <= self.config.cell_size_m / 2.0) | (np.abs(errors) <= half_fov)
        )
        altitude = self.scene.flight_altitude_m
        origin = np.asarray((position[0], position[1], altitude), dtype=np.float32)
        for index in np.flatnonzero(candidates):
            center = self._free_centers[index]
            target = np.asarray((center[0], center[1], altitude), dtype=np.float32)
            if line_of_sight_clear(
                self.scene,
                origin,
                target,
                ignored_object_id="",
            ):
                cell = tuple(int(value) for value in self._free_indices[index])
                self.visible[cell] = True
