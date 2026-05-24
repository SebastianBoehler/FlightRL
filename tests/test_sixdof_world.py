from __future__ import annotations

import numpy as np

from flightrl.sixdof import AxisAlignedObstacle, BoxRoom


def test_box_room_raycast_hits_internal_obstacle_before_wall() -> None:
    room = BoxRoom(obstacles=(AxisAlignedObstacle(x_min=0.8, x_max=1.0, y_min=-0.2, y_max=0.2, z_min=0.0, z_max=1.0),))
    position = np.asarray([[0.0, 0.0, 0.4]], dtype=np.float32)
    direction = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    distance = room.raycast(position, direction)

    np.testing.assert_allclose(distance, [0.8], atol=1e-5)


def test_box_room_contains_rejects_obstacle_interior() -> None:
    room = BoxRoom(obstacles=(AxisAlignedObstacle(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z_min=0.0, z_max=1.0),))
    positions = np.asarray([[0.0, 0.0, 0.5], [0.5, 0.0, 0.5]], dtype=np.float32)

    contained = room.contains(positions, margin=0.03)

    assert contained.tolist() == [False, True]
