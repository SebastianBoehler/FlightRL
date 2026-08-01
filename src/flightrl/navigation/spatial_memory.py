from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .semantic_scene import Bounds3D


MAP_CHANNELS = ("visited", "observed_free", "occupied", "target_evidence", "agent")


@dataclass(frozen=True, slots=True)
class SpatialMemoryConfig:
    cell_size_m: float = 0.25
    local_size: int = 16
    visited_radius_m: float = 0.18

    def __post_init__(self) -> None:
        if self.cell_size_m <= 0.0:
            raise ValueError("map cell size must be positive")
        if self.local_size <= 0 or self.local_size % 2:
            raise ValueError("local map size must be a positive even number")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(MAP_CHANNELS), self.local_size, self.local_size)

    @property
    def flat_dim(self) -> int:
        return int(np.prod(self.shape))


class EgocentricSpatialMemory:
    """Persistent world map exposed as a small body-aligned local crop."""

    def __init__(
        self,
        room: Bounds3D,
        config: SpatialMemoryConfig | None = None,
    ) -> None:
        self.room = room
        self.config = config or SpatialMemoryConfig()
        width = room.maximum[0] - room.minimum[0]
        height = room.maximum[1] - room.minimum[1]
        columns = int(np.ceil(width / self.config.cell_size_m)) + 1
        rows = int(np.ceil(height / self.config.cell_size_m)) + 1
        self.grid = np.zeros((len(MAP_CHANNELS) - 1, rows, columns), dtype=np.float32)

    def reset(self) -> None:
        self.grid.fill(0.0)

    def update_pose(self, position_xy: np.ndarray) -> int:
        before = int(np.count_nonzero(self.grid[0]))
        self._mark_disk(
            0,
            np.asarray(position_xy, dtype=np.float32),
            self.config.visited_radius_m,
            1.0,
        )
        return int(np.count_nonzero(self.grid[0])) - before

    def observe_rays(
        self,
        position_xy: np.ndarray,
        yaw_rad: float,
        bearings_rad: np.ndarray,
        ranges_m: np.ndarray,
        *,
        max_range_m: float,
    ) -> None:
        origin = np.asarray(position_xy, dtype=np.float32)
        for bearing, distance in zip(bearings_rad, ranges_m, strict=True):
            clipped = float(np.clip(distance, 0.0, max_range_m))
            angle = float(yaw_rad + bearing)
            direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
            for sample in np.arange(
                self.config.cell_size_m,
                clipped,
                self.config.cell_size_m,
            ):
                self._mark_cell(1, origin + direction * sample, 1.0)
            if clipped < max_range_m:
                self._mark_cell(2, origin + direction * clipped, 1.0)

    def observe_semantic(
        self,
        position_xy: np.ndarray,
        yaw_rad: float,
        bearing_rad: float,
        distance_m: float,
        confidence: float,
        *,
        replace: bool = False,
    ) -> None:
        if replace:
            self.grid[3].fill(0.0)
        angle = float(yaw_rad + bearing_rad)
        endpoint = np.asarray(position_xy, dtype=np.float32) + distance_m * np.asarray(
            [np.cos(angle), np.sin(angle)],
            dtype=np.float32,
        )
        self._mark_disk(
            3, endpoint, self.config.cell_size_m, float(np.clip(confidence, 0.0, 1.0))
        )

    def local_map(self, position_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
        size = self.config.local_size
        center = (size - 1) / 2.0
        rows, columns = np.meshgrid(
            np.arange(size, dtype=np.float32),
            np.arange(size, dtype=np.float32),
            indexing="ij",
        )
        forward = (center - rows) * self.config.cell_size_m
        left = (center - columns) * self.config.cell_size_m
        cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
        origin = np.asarray(position_xy, dtype=np.float32)
        world_x = origin[0] + cosine * forward - sine * left
        world_y = origin[1] + sine * forward + cosine * left
        grid_rows, grid_columns, valid = self._world_to_cells(world_x, world_y)
        local = np.zeros(self.config.shape, dtype=np.float32)
        for channel in range(self.grid.shape[0]):
            local[channel, valid] = self.grid[
                channel, grid_rows[valid], grid_columns[valid]
            ]
        local[-1, size // 2, size // 2] = 1.0
        return local

    def _mark_disk(
        self,
        channel: int,
        position_xy: np.ndarray,
        radius_m: float,
        value: float,
    ) -> None:
        row, column, valid = self._world_to_cells(position_xy[0], position_xy[1])
        if not bool(valid):
            return
        radius = max(0, int(np.ceil(radius_m / self.config.cell_size_m)))
        for row_offset in range(-radius, radius + 1):
            for column_offset in range(-radius, radius + 1):
                if (
                    row_offset * row_offset + column_offset * column_offset
                    > radius * radius
                ):
                    continue
                selected_row = int(row) + row_offset
                selected_column = int(column) + column_offset
                if (
                    0 <= selected_row < self.grid.shape[1]
                    and 0 <= selected_column < self.grid.shape[2]
                ):
                    self.grid[channel, selected_row, selected_column] = max(
                        self.grid[channel, selected_row, selected_column],
                        value,
                    )

    def _mark_cell(self, channel: int, position_xy: np.ndarray, value: float) -> None:
        row, column, valid = self._world_to_cells(position_xy[0], position_xy[1])
        if bool(valid):
            self.grid[channel, int(row), int(column)] = value

    def _world_to_cells(self, x, y):
        columns = np.rint(
            (np.asarray(x) - self.room.minimum[0]) / self.config.cell_size_m
        ).astype(np.intp)
        rows = np.rint(
            (np.asarray(y) - self.room.minimum[1]) / self.config.cell_size_m
        ).astype(np.intp)
        valid = (
            (rows >= 0)
            & (rows < self.grid.shape[1])
            & (columns >= 0)
            & (columns < self.grid.shape[2])
        )
        return rows, columns, valid
