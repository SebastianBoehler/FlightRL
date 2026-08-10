from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import cos, floor, isfinite, pi, sin

import numpy as np

from .range_contract import RANGE_MAP_SHAPE


_BEARINGS = (0.0, pi, pi / 2.0, -pi / 2.0)
_FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_EIGHT_NEIGHBORS = tuple(
    (column, row)
    for column in (-1, 0, 1)
    for row in (-1, 0, 1)
    if column or row
)


@dataclass(frozen=True, slots=True)
class RangePose:
    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x_m, self.y_m, self.yaw_rad)):
            raise ValueError("range pose values must be finite")


class RangeOccupancyMap:
    def __init__(
        self,
        *,
        cell_size_m: float = 0.10,
        max_range_m: float = 4.0,
    ) -> None:
        if not 0.0 < cell_size_m < max_range_m:
            raise ValueError("range map cell size and maximum range are invalid")
        self.cell_size_m = float(cell_size_m)
        self.max_range_m = float(max_range_m)
        self.log_odds: dict[tuple[int, int], float] = {}
        self.visited: set[tuple[int, int]] = set()

    def reset(self) -> None:
        self.log_odds.clear()
        self.visited.clear()

    def update(
        self,
        pose: RangePose,
        ranges_m: np.ndarray,
        validity: np.ndarray,
        *,
        visited: bool = True,
    ) -> None:
        ranges = np.asarray(ranges_m, dtype=np.float32)
        valid = np.asarray(validity, dtype=np.float32)
        if ranges.shape != (4,) or valid.shape != (4,):
            raise ValueError("range map update requires four ranges and validity flags")
        if not np.isfinite(ranges).all() or not np.isfinite(valid).all():
            raise ValueError("range map update values must be finite")
        if np.any((valid != 0.0) & (valid != 1.0)):
            raise ValueError("range map validity values must be binary")
        if visited:
            self.visited.add(self._cell(pose.x_m, pose.y_m))
        for bearing, distance_value, is_valid in zip(
            _BEARINGS, ranges, valid, strict=True
        ):
            distance = float(distance_value)
            finite_return = bool(is_valid)
            if finite_return:
                if not 0.03 <= distance <= self.max_range_m:
                    raise ValueError("finite ranger return is outside [0.03, 4.0]m")
            elif distance != self.max_range_m:
                raise ValueError("no-return ranger value must equal maximum range")
            cells = self._ray_cells(pose, bearing, distance)
            free_cells = cells[:-1] if finite_return and cells else cells
            for cell in free_cells:
                self._add_log_odds(cell, -0.4)
            if finite_return and cells:
                self._add_log_odds(cells[-1], 0.85)

    def cell_state(self, cell: tuple[int, int]) -> str:
        value = self.log_odds.get(cell, 0.0)
        if value <= -0.8:
            return "free"
        if value >= 1.7:
            return "occupied"
        if cell in self.visited:
            return "visited"
        return "unknown"

    def frontier_cells(self, pose: RangePose) -> set[tuple[int, int]]:
        free = {cell for cell, value in self.log_odds.items() if value <= -0.8}
        traversable = free | self.visited
        start = self._cell(pose.x_m, pose.y_m)
        if start not in traversable:
            return set()
        reachable = self._reachable(start, traversable)
        candidates = {
            cell
            for cell in reachable & free
            if any(
                self.cell_state((cell[0] + dc, cell[1] + dr)) == "unknown"
                for dc, dr in _FOUR_NEIGHBORS
            )
        }
        retained: set[tuple[int, int]] = set()
        remaining = set(candidates)
        while remaining:
            seed = remaining.pop()
            cluster = {seed}
            queue = [seed]
            while queue:
                current = queue.pop()
                for dc, dr in _EIGHT_NEIGHBORS:
                    neighbor = (current[0] + dc, current[1] + dr)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        cluster.add(neighbor)
                        queue.append(neighbor)
            if len(cluster) >= 3:
                retained.update(cluster)
        return retained

    def exploration_crop(self, pose: RangePose) -> np.ndarray:
        output = np.zeros(RANGE_MAP_SHAPE, dtype=np.float32)
        frontiers = self.frontier_cells(pose)
        center = RANGE_MAP_SHAPE[1] // 2
        cosine, sine = cos(pose.yaw_rad), sin(pose.yaw_rad)
        for row in range(RANGE_MAP_SHAPE[1]):
            forward = (center - row) * 0.20
            for column in range(RANGE_MAP_SHAPE[2]):
                left = (center - column) * 0.20
                world_x = pose.x_m + cosine * forward - sine * left
                world_y = pose.y_m + sine * forward + cosine * left
                cell = self._cell(world_x, world_y)
                state = self.cell_state(cell)
                output[0, row, column] = float(cell in self.visited)
                output[1, row, column] = float(state == "free")
                output[2, row, column] = float(state == "occupied")
                output[3, row, column] = float(cell in frontiers)
        return output

    def _add_log_odds(self, cell: tuple[int, int], delta: float) -> None:
        self.log_odds[cell] = float(np.clip(self.log_odds.get(cell, 0.0) + delta, -2.0, 3.0))

    def _cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        return floor(x_m / self.cell_size_m), floor(y_m / self.cell_size_m)

    def _ray_cells(
        self,
        pose: RangePose,
        bearing: float,
        distance_m: float,
    ) -> list[tuple[int, int]]:
        angle = pose.yaw_rad + bearing
        samples = max(1, int(np.ceil(distance_m / (self.cell_size_m * 0.25))))
        origin = self._cell(pose.x_m, pose.y_m)
        cells: list[tuple[int, int]] = []
        for index in range(1, samples + 1):
            distance = distance_m * index / samples
            cell = self._cell(
                pose.x_m + cos(angle) * distance,
                pose.y_m + sin(angle) * distance,
            )
            if cell != origin and (not cells or cells[-1] != cell):
                cells.append(cell)
        return cells

    @staticmethod
    def _reachable(
        start: tuple[int, int],
        traversable: set[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        reached = {start}
        queue = deque((start,))
        while queue:
            current = queue.popleft()
            for dc, dr in _FOUR_NEIGHBORS:
                neighbor = (current[0] + dc, current[1] + dr)
                if neighbor in traversable and neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        return reached
