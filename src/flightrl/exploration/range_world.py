from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import cos, floor, pi, sin

import numpy as np

from .range_mapper import RangePose


_BEARINGS = (0.0, pi, pi / 2.0, -pi / 2.0)


@dataclass(frozen=True, slots=True)
class RangeWorld:
    occupied: np.ndarray
    cell_size_m: float = 0.10

    def __post_init__(self) -> None:
        grid = np.asarray(self.occupied, dtype=bool)
        if grid.shape != (64, 64):
            raise ValueError("range world must be a 64 by 64 occupancy grid")
        if not np.all(grid[[0, -1], :]) or not np.all(grid[:, [0, -1]]):
            raise ValueError("range world boundary must be occupied")
        object.__setattr__(self, "occupied", grid.copy())

    @classmethod
    def open_room(cls) -> "RangeWorld":
        grid = np.zeros((64, 64), dtype=bool)
        grid[[0, -1], :] = True
        grid[:, [0, -1]] = True
        return cls(grid)

    @classmethod
    def generate(cls, seed: int) -> "RangeWorld":
        rng = np.random.default_rng(seed)
        grid = cls.open_room().occupied.copy()
        accepted = 0
        attempts = 0
        while accepted < 6 and attempts < 100:
            attempts += 1
            height, width = rng.integers(3, 10, size=2)
            row = int(rng.integers(3, 61 - height))
            column = int(rng.integers(3, 61 - width))
            candidate = grid.copy()
            candidate[row : row + height, column : column + width] = True
            world = cls(candidate)
            if world.free_space_is_connected():
                grid = candidate
                accepted += 1
        return cls(grid)

    @property
    def free_cell_count(self) -> int:
        return int(np.count_nonzero(~self.occupied))

    def free_space_is_connected(self) -> bool:
        free = np.argwhere(~self.occupied)
        if len(free) == 0:
            return False
        start = tuple(int(value) for value in free[0])
        reached = {start}
        queue = deque((start,))
        while queue:
            row, column = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (row + dr, column + dc)
                if (
                    0 <= neighbor[0] < 64
                    and 0 <= neighbor[1] < 64
                    and not self.occupied[neighbor]
                    and neighbor not in reached
                ):
                    reached.add(neighbor)
                    queue.append(neighbor)
        return len(reached) == len(free)

    def sample_pose(self, rng: np.random.Generator) -> RangePose:
        candidates = np.argwhere(~self.occupied)
        for index in rng.permutation(len(candidates)):
            row, column = candidates[int(index)]
            pose = RangePose(
                (float(column) + 0.5) * self.cell_size_m,
                (float(row) + 0.5) * self.cell_size_m,
                float(rng.uniform(-pi, pi)),
            )
            ranges, validity = self.horizontal_ranges(pose)
            finite = ranges[validity.astype(bool)]
            if (
                not self.collides(pose.x_m, pose.y_m)
                and (len(finite) == 0 or float(np.min(finite)) >= 0.35)
            ):
                return pose
        raise RuntimeError("range world contains no collision-free start pose")

    def horizontal_ranges(
        self,
        pose: RangePose,
        *,
        max_range_m: float = 4.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        ranges = np.empty(4, dtype=np.float32)
        validity = np.ones(4, dtype=np.float32)
        for index, bearing in enumerate(_BEARINGS):
            distance = self._ray_distance(pose, bearing, max_range_m)
            ranges[index] = distance
            if distance >= max_range_m:
                validity[index] = 0.0
        return ranges, validity

    def visible_free_cells(
        self,
        pose: RangePose,
        *,
        max_range_m: float = 4.0,
    ) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for bearing in _BEARINGS:
            angle = pose.yaw_rad + bearing
            for distance in np.arange(0.0, max_range_m + 0.005, 0.02):
                x_m = pose.x_m + cos(angle) * float(distance)
                y_m = pose.y_m + sin(angle) * float(distance)
                cell = self._cell(x_m, y_m)
                if cell is None or self.occupied[cell]:
                    break
                cells.add(cell)
        return cells

    def collides(self, x_m: float, y_m: float, *, radius_m: float = 0.15) -> bool:
        for angle in np.linspace(0.0, 2.0 * pi, 24, endpoint=False):
            cell = self._cell(
                x_m + cos(float(angle)) * radius_m,
                y_m + sin(float(angle)) * radius_m,
            )
            if cell is None or self.occupied[cell]:
                return True
        center = self._cell(x_m, y_m)
        return center is None or bool(self.occupied[center])

    def truth_cell(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        return self._cell(x_m, y_m)

    def _ray_distance(self, pose: RangePose, bearing: float, maximum: float) -> float:
        angle = pose.yaw_rad + bearing
        for distance in np.arange(0.0, maximum + 0.005, 0.01):
            cell = self._cell(
                pose.x_m + cos(angle) * float(distance),
                pose.y_m + sin(angle) * float(distance),
            )
            if cell is None or self.occupied[cell]:
                return float(distance)
        return maximum

    def _cell(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        column = floor(x_m / self.cell_size_m)
        row = floor(y_m / self.cell_size_m)
        if not 0 <= row < 64 or not 0 <= column < 64:
            return None
        return row, column
