from __future__ import annotations

from math import pi

import numpy as np

from flightrl.exploration.range_mapper import RangeOccupancyMap, RangePose


FOUR_RETURNS = np.asarray((0.4, 0.4, 0.4, 0.4), dtype=np.float32)
ALL_VALID = np.ones(4, dtype=np.float32)


def test_repeated_finite_ray_marks_free_traversal_and_occupied_endpoint() -> None:
    mapper = RangeOccupancyMap()
    pose = RangePose(0.05, 0.05, 0.0)
    forward_only = np.asarray((0.4, 4.0, 4.0, 4.0), dtype=np.float32)
    validity = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)

    mapper.update(pose, forward_only, validity)
    assert mapper.cell_state((2, 0)) == "unknown"
    assert mapper.cell_state((4, 0)) == "unknown"

    mapper.update(pose, forward_only, validity)

    assert mapper.cell_state((1, 0)) == "free"
    assert mapper.cell_state((2, 0)) == "free"
    assert mapper.cell_state((3, 0)) == "free"
    assert mapper.cell_state((4, 0)) == "occupied"
    assert mapper.cell_state((0, 0)) == "visited"


def test_no_return_clears_ray_without_creating_four_meter_wall() -> None:
    mapper = RangeOccupancyMap()
    pose = RangePose(0.05, 0.05, 0.0)
    no_returns = np.full(4, 4.0, dtype=np.float32)
    invalid = np.zeros(4, dtype=np.float32)

    mapper.update(pose, no_returns, invalid)
    mapper.update(pose, no_returns, invalid)

    assert mapper.cell_state((20, 0)) == "free"
    assert mapper.cell_state((40, 0)) != "occupied"


def test_frontier_extractor_exposes_all_reachable_clusters() -> None:
    mapper = RangeOccupancyMap()
    pose = RangePose(0.05, 0.05, 0.0)

    mapper.update(pose, FOUR_RETURNS, ALL_VALID)
    mapper.update(pose, FOUR_RETURNS, ALL_VALID)
    frontiers = mapper.frontier_cells(pose)

    assert len(frontiers) >= 8
    assert any(column > 0 and row == 0 for column, row in frontiers)
    assert any(column < 0 and row == 0 for column, row in frontiers)
    assert any(row > 0 and column == 0 for column, row in frontiers)
    assert any(row < 0 and column == 0 for column, row in frontiers)


def test_body_aligned_crop_rotates_world_obstacle_with_drone_yaw() -> None:
    mapper = RangeOccupancyMap()
    origin = RangePose(0.05, 0.05, 0.0)
    forward_only = np.asarray((0.4, 4.0, 4.0, 4.0), dtype=np.float32)
    validity = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    mapper.update(origin, forward_only, validity)
    mapper.update(origin, forward_only, validity)

    facing_obstacle = mapper.exploration_crop(origin)
    obstacle_to_right = mapper.exploration_crop(RangePose(0.05, 0.05, pi / 2.0))

    assert facing_obstacle[2, 14, 16] == np.float32(1.0)
    assert obstacle_to_right[2, 16, 18] == np.float32(1.0)
    assert facing_obstacle.shape == (4, 32, 32)


def test_reset_removes_map_and_frontier_state() -> None:
    mapper = RangeOccupancyMap()
    pose = RangePose(0.05, 0.05, 0.0)
    mapper.update(pose, FOUR_RETURNS, ALL_VALID)
    mapper.update(pose, FOUR_RETURNS, ALL_VALID)

    mapper.reset()

    assert mapper.frontier_cells(pose) == set()
    assert np.count_nonzero(mapper.exploration_crop(pose)) == 0
