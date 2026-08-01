from __future__ import annotations

import xml.etree.ElementTree as ET

from flightrl.mujoco.model import build_crazyflie_mjcf
from flightrl.navigation import (
    SEMANTIC_TARGET_CATEGORIES,
    SemanticRoomGenerationConfig,
    generate_semantic_room,
)


def test_semantic_room_generation_is_seeded_and_contains_targets() -> None:
    first = generate_semantic_room(71)
    repeated = generate_semantic_room(71)
    different = generate_semantic_room(72)

    assert first == repeated
    assert first != different
    for category in SEMANTIC_TARGET_CATEGORIES:
        _, target = first.object_by_name(category)
        assert first.room.contains(target.bounds.minimum)
        assert first.room.contains(target.bounds.maximum)
        assert target.approach_position_m is not None
        assert first.room.contains(
            target.approach_position_m,
            margin=first.boundary_margin_m,
        )


def test_generated_room_resizes_mujoco_geometry_and_removes_markers() -> None:
    scene = generate_semantic_room(19)
    root = ET.fromstring(build_crazyflie_mjcf(scene))

    ceiling = root.find(".//geom[@name='ceiling']")
    wall = root.find(".//geom[@name='wall_x_pos']")

    assert ceiling is not None
    assert wall is not None
    assert float(ceiling.attrib["pos"].split()[2]) > scene.room.maximum[2]
    assert float(wall.attrib["pos"].split()[0]) > scene.room.maximum[0]
    assert root.find(".//geom[@name='marker_x']") is None
    assert root.find(".//geom[@name='semantic_oven_0']") is not None


def test_diverse_rooms_randomize_appearance_and_obstacle_geometry() -> None:
    config = SemanticRoomGenerationConfig.for_profile("diverse")
    scenes = tuple(generate_semantic_room(seed, config) for seed in range(20, 32))

    assert len({scene.appearance for scene in scenes}) == len(scenes)
    shapes = {
        obj.shape
        for scene in scenes
        for obj in scene.objects
        if obj.category == "obstacle"
    }
    assert shapes == {"box", "cylinder"}
    assert all(
        1 <= sum(obj.category == "obstacle" for obj in scene.objects) <= 6
        for scene in scenes
    )


def test_generated_room_adds_seeded_surface_materials() -> None:
    scene = generate_semantic_room(
        91,
        SemanticRoomGenerationConfig.for_profile("diverse"),
    )
    root = ET.fromstring(build_crazyflie_mjcf(scene))

    assert root.find(".//texture[@name='semantic_floor_texture']") is not None
    assert root.find(".//texture[@name='semantic_wall_texture']") is not None
    assert (
        root.find(".//geom[@name='floor']").attrib["material"]
        == "semantic_floor_material"
    )
    assert (
        root.find(".//geom[@name='wall_x_pos']").attrib["material"]
        == "semantic_wall_material"
    )
