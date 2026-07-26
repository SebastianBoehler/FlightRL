from __future__ import annotations

import xml.etree.ElementTree as ET

from flightrl.navigation.semantic_scene import SemanticScene
from flightrl.sixdof.geometry import AxisAlignedObstacle, BoxRoom


def add_semantic_scene_to_mjcf(mjcf: str, scene: SemanticScene) -> str:
    """Add named semantic box geometry while preserving the base drone model."""
    root = ET.fromstring(mjcf)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo model has no worldbody")

    for obj in scene.objects:
        attributes = {
            "name": f"semantic_{obj.object_id}",
            "type": "box",
            "pos": _vector(obj.bounds.center),
            "size": _vector(obj.bounds.half_extents),
            "rgba": _vector(obj.rgba),
            "contype": "1" if obj.collision else "0",
            "conaffinity": "1" if obj.collision else "0",
        }
        ET.SubElement(worldbody, "geom", attributes)
    return ET.tostring(root, encoding="unicode")


def box_room_from_semantic_scene(scene: SemanticScene, max_range_m: float = 4.0) -> BoxRoom:
    colliders = tuple(
        AxisAlignedObstacle(
            x_min=obj.bounds.minimum[0],
            x_max=obj.bounds.maximum[0],
            y_min=obj.bounds.minimum[1],
            y_max=obj.bounds.maximum[1],
            z_min=obj.bounds.minimum[2],
            z_max=obj.bounds.maximum[2],
        )
        for obj in scene.objects
        if obj.collision
    )
    return BoxRoom(
        x_min=scene.room.minimum[0],
        x_max=scene.room.maximum[0],
        y_min=scene.room.minimum[1],
        y_max=scene.room.maximum[1],
        z_min=scene.room.minimum[2],
        z_max=scene.room.maximum[2],
        max_range_m=max_range_m,
        obstacles=colliders,
    )


def _vector(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:g}" for value in values)
