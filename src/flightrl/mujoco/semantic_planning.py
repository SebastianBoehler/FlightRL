from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import hypot

import numpy as np

from flightrl.navigation.semantic_scene import SemanticScene


GridCell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class PrivilegedGridConfig:
    cell_size_m: float = 0.25
    boundary_margin_m: float = 0.35
    obstacle_inflation_m: float = 0.28
    coverage_margin_m: float = 0.75


class PrivilegedGridPlanner:
    """Collision-aware teacher planner whose state is never exposed to the actor."""

    def __init__(
        self,
        scene: SemanticScene,
        config: PrivilegedGridConfig | None = None,
    ) -> None:
        self.scene = scene
        self.config = config or PrivilegedGridConfig()
        room = scene.room
        margin = self.config.boundary_margin_m
        self.xs = np.arange(
            room.minimum[0] + margin,
            room.maximum[0] - margin + 1e-6,
            self.config.cell_size_m,
            dtype=np.float32,
        )
        self.ys = np.arange(
            room.minimum[1] + margin,
            room.maximum[1] - margin + 1e-6,
            self.config.cell_size_m,
            dtype=np.float32,
        )
        self.blocked = np.zeros((len(self.xs), len(self.ys)), dtype=bool)
        self._mark_obstacles()

    def path(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
    ) -> list[np.ndarray]:
        start = self.nearest_free_cell(start_xy)
        goal = self.nearest_free_cell(goal_xy)
        frontier: list[tuple[float, float, GridCell]] = [(0.0, 0.0, start)]
        previous: dict[GridCell, GridCell] = {}
        costs = {start: 0.0}
        while frontier:
            _priority, cost, current = heapq.heappop(frontier)
            if current == goal:
                return self._reconstruct(previous, current)
            if cost > costs[current]:
                continue
            for neighbor, step_cost in self._neighbors(current):
                candidate = cost + step_cost
                if candidate >= costs.get(neighbor, float("inf")):
                    continue
                costs[neighbor] = candidate
                previous[neighbor] = current
                priority = candidate + hypot(
                    neighbor[0] - goal[0],
                    neighbor[1] - goal[1],
                )
                heapq.heappush(frontier, (priority, candidate, neighbor))
        return []

    def coverage_goals(self) -> tuple[np.ndarray, ...]:
        room = self.scene.room
        margin = self.config.coverage_margin_m
        requested = (
            (0.0, 0.0),
            (room.minimum[0] + margin, room.minimum[1] + margin),
            (room.maximum[0] - margin, room.minimum[1] + margin),
            (room.maximum[0] - margin, room.maximum[1] - margin),
            (room.minimum[0] + margin, room.maximum[1] - margin),
        )
        return tuple(
            self.cell_center(self._nearest_coverage_cell(point)) for point in requested
        )

    def nearest_free_cell(self, point_xy) -> GridCell:
        point = np.asarray(point_xy, dtype=np.float32)
        x_index = int(np.argmin(np.abs(self.xs - point[0])))
        y_index = int(np.argmin(np.abs(self.ys - point[1])))
        if not self.blocked[x_index, y_index]:
            return x_index, y_index
        free = np.argwhere(~self.blocked)
        if len(free) == 0:
            raise RuntimeError("privileged planner has no free cells")
        centers = np.column_stack((self.xs[free[:, 0]], self.ys[free[:, 1]]))
        nearest = free[int(np.argmin(np.sum((centers - point) ** 2, axis=1)))]
        return int(nearest[0]), int(nearest[1])

    def cell_center(self, cell: GridCell) -> np.ndarray:
        return np.asarray((self.xs[cell[0]], self.ys[cell[1]]), dtype=np.float32)

    def _nearest_coverage_cell(self, point_xy) -> GridCell:
        margin = self.config.coverage_margin_m
        room = self.scene.room
        interior_x = (self.xs >= room.minimum[0] + margin) & (
            self.xs <= room.maximum[0] - margin
        )
        interior_y = (self.ys >= room.minimum[1] + margin) & (
            self.ys <= room.maximum[1] - margin
        )
        candidates = np.argwhere(
            ~self.blocked & np.logical_and.outer(interior_x, interior_y)
        )
        if len(candidates) == 0:
            return self.nearest_free_cell(point_xy)
        point = np.asarray(point_xy, dtype=np.float32)
        centers = np.column_stack(
            (self.xs[candidates[:, 0]], self.ys[candidates[:, 1]])
        )
        nearest = candidates[int(np.argmin(np.sum((centers - point) ** 2, axis=1)))]
        return int(nearest[0]), int(nearest[1])

    def _mark_obstacles(self) -> None:
        altitude = self.scene.flight_altitude_m
        inflation = self.config.obstacle_inflation_m
        for obj in self.scene.objects:
            if (
                not obj.collision
                or obj.bounds.maximum[2] < altitude - 0.15
                or obj.bounds.minimum[2] > altitude + 0.15
            ):
                continue
            x_mask = (self.xs >= obj.bounds.minimum[0] - inflation) & (
                self.xs <= obj.bounds.maximum[0] + inflation
            )
            y_mask = (self.ys >= obj.bounds.minimum[1] - inflation) & (
                self.ys <= obj.bounds.maximum[1] + inflation
            )
            self.blocked[np.ix_(x_mask, y_mask)] = True

    def _neighbors(self, cell: GridCell):
        x, y = cell
        for dx, dy in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ):
            candidate = x + dx, y + dy
            if (
                candidate[0] < 0
                or candidate[0] >= len(self.xs)
                or candidate[1] < 0
                or candidate[1] >= len(self.ys)
                or self.blocked[candidate]
            ):
                continue
            if dx and dy and (self.blocked[x + dx, y] or self.blocked[x, y + dy]):
                continue
            yield candidate, hypot(dx, dy)

    def _reconstruct(
        self,
        previous: dict[GridCell, GridCell],
        current: GridCell,
    ) -> list[np.ndarray]:
        cells = [current]
        while current in previous:
            current = previous[current]
            cells.append(current)
        cells.reverse()
        return [self.cell_center(cell) for cell in cells]
