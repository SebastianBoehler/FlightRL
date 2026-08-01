from __future__ import annotations

import numpy as np

from flightrl.mujoco.semantic_exploration import line_of_sight_clear
from flightrl.mujoco.semantic_planning import PrivilegedGridPlanner
from flightrl.navigation.mission_spec import TargetAnchor
from flightrl.navigation.room_generation import (
    SemanticRoomGenerationConfig,
    generate_semantic_room,
)
from flightrl.navigation.semantic_scene import Bounds3D, SemanticObject, SemanticScene


def test_room_generation_adds_seeded_interior_obstacles() -> None:
    config = SemanticRoomGenerationConfig(obstacle_count_range=(3, 3))

    scene = generate_semantic_room(81, config)
    obstacles = [obj for obj in scene.objects if obj.category == "obstacle"]

    assert scene == generate_semantic_room(81, config)
    assert len(obstacles) == 3
    assert all(obj.collision for obj in obstacles)


def test_privileged_planner_routes_around_inflated_obstacle() -> None:
    scene = generate_semantic_room(
        84,
        SemanticRoomGenerationConfig(obstacle_count_range=(3, 3)),
    )
    planner = PrivilegedGridPlanner(scene)

    path = planner.path(
        np.asarray((-1.5, -1.5), dtype=np.float32),
        np.asarray((1.5, 1.5), dtype=np.float32),
    )

    assert len(path) > 1
    assert all(not planner.blocked[planner.nearest_free_cell(point)] for point in path)


def test_privileged_coverage_goals_are_free_interior_viewpoints() -> None:
    scene = generate_semantic_room(
        86,
        SemanticRoomGenerationConfig.for_profile("diverse"),
    )
    planner = PrivilegedGridPlanner(scene)

    goals = planner.coverage_goals()

    assert len(goals) == 5
    assert all(not planner.blocked[planner.nearest_free_cell(goal)] for goal in goals)
    assert all(scene.room.minimum[0] + 0.5 < goal[0] for goal in goals)
    assert all(goal[0] < scene.room.maximum[0] - 0.5 for goal in goals)


def test_collision_geometry_occludes_semantic_target() -> None:
    target = SemanticObject(
        "monitor_0",
        "monitor",
        Bounds3D((2.0, -0.3, 0.7), (2.1, 0.3, 1.4)),
        preferred_anchor=TargetAnchor.APPROACH,
        approach_position_m=(1.5, 0.0, 1.0),
        collision=False,
    )
    blocker = SemanticObject(
        "obstacle_0",
        "obstacle",
        Bounds3D((0.8, -0.4, 0.0), (1.2, 0.4, 1.8)),
        preferred_anchor=TargetAnchor.CENTER,
        collision=True,
    )
    scene = SemanticScene(
        Bounds3D((-2.0, -2.0, 0.0), (3.0, 2.0, 2.5)),
        (target, blocker),
        flight_altitude_m=1.0,
    )

    assert not line_of_sight_clear(
        scene,
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray(target.bounds.center),
        ignored_object_id=target.object_id,
    )
