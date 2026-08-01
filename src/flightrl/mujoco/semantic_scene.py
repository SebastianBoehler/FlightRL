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

    _add_room_appearance(root, worldbody, scene)
    _resize_room(worldbody, scene)
    for name in ("marker_x", "marker_y"):
        marker = worldbody.find(f"geom[@name='{name}']")
        if marker is not None:
            worldbody.remove(marker)

    for obj in scene.objects:
        attributes = {
            "name": f"semantic_{obj.object_id}",
            "type": obj.shape,
            "pos": _vector(obj.bounds.center),
            "size": _object_size(obj.shape, obj.bounds.half_extents),
            "rgba": _vector(obj.rgba),
            "contype": "1" if obj.collision else "0",
            "conaffinity": "1" if obj.collision else "0",
        }
        ET.SubElement(worldbody, "geom", attributes)
        if obj.category in {"door", "doorway"}:
            _add_door_structure(worldbody, obj)
    return ET.tostring(root, encoding="unicode")


def _add_room_appearance(
    root: ET.Element,
    worldbody: ET.Element,
    scene: SemanticScene,
) -> None:
    asset = root.find("asset")
    if asset is None:
        raise ValueError("MuJoCo model has no asset section")
    appearance = scene.appearance
    for name, first, second in (
        (
            "semantic_floor_texture",
            appearance.floor_rgb1,
            appearance.floor_rgb2,
        ),
        (
            "semantic_wall_texture",
            appearance.wall_rgb1,
            appearance.wall_rgb2,
        ),
    ):
        ET.SubElement(
            asset,
            "texture",
            {
                "name": name,
                "type": "2d",
                "builtin": "checker",
                "width": "64",
                "height": "64",
                "rgb1": _vector(first),
                "rgb2": _vector(second),
            },
        )
    for surface in ("floor", "wall"):
        ET.SubElement(
            asset,
            "material",
            {
                "name": f"semantic_{surface}_material",
                "texture": f"semantic_{surface}_texture",
                "texrepeat": (
                    f"{appearance.checker_repeat:g} "
                    f"{appearance.checker_repeat:g}"
                ),
                "texuniform": "true",
            },
        )
    floor = worldbody.find("geom[@name='floor']")
    if floor is not None:
        floor.set("material", "semantic_floor_material")
    for name in ("ceiling", "wall_x_neg", "wall_x_pos", "wall_y_neg", "wall_y_pos"):
        geom = worldbody.find(f"geom[@name='{name}']")
        if geom is not None:
            geom.set("material", "semantic_wall_material")


def _resize_room(worldbody: ET.Element, scene: SemanticScene) -> None:
    x_min, y_min, z_min = scene.room.minimum
    x_max, y_max, z_max = scene.room.maximum
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    center_z = 0.5 * (z_min + z_max)
    half_x = 0.5 * (x_max - x_min)
    half_y = 0.5 * (y_max - y_min)
    half_z = 0.5 * (z_max - z_min)
    room_geometry = {
        "floor": ((center_x, center_y, z_min), (half_x, half_y, 0.05)),
        "ceiling": ((center_x, center_y, z_max + 0.02), (half_x, half_y, 0.02)),
        "wall_x_neg": ((x_min - 0.02, center_y, center_z), (0.02, half_y, half_z)),
        "wall_x_pos": ((x_max + 0.02, center_y, center_z), (0.02, half_y, half_z)),
        "wall_y_neg": ((center_x, y_min - 0.02, center_z), (half_x, 0.02, half_z)),
        "wall_y_pos": ((center_x, y_max + 0.02, center_z), (half_x, 0.02, half_z)),
    }
    for name, (position, size) in room_geometry.items():
        geom = worldbody.find(f"geom[@name='{name}']")
        if geom is None:
            raise ValueError(f"MuJoCo base model is missing room geometry {name!r}")
        geom.set("pos", _vector(position))
        geom.set("size", _vector(size))


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


def _object_size(shape: str, half_extents: tuple[float, float, float]) -> str:
    if shape == "cylinder":
        return _vector((max(half_extents[0], half_extents[1]), half_extents[2]))
    return _vector(half_extents)


def _add_door_structure(worldbody: ET.Element, obj) -> None:
    center = obj.bounds.center
    half = obj.bounds.half_extents
    normal_axis = 0 if half[0] < half[1] else 1
    tangent_axis = 1 - normal_axis
    surface = list(center)
    direction_to_room = -1.0 if center[normal_axis] > 0.0 else 1.0
    surface[normal_axis] += direction_to_room * (half[normal_axis] + 0.008)
    frame_width = min(0.065, 0.10 * half[tangent_axis])
    depth_half = 0.006
    frame_color = _contrast_color(obj.rgba, 0.34)
    panel_color = _contrast_color(obj.rgba, -0.20)
    handle_color = (0.78, 0.74, 0.52, 1.0)
    prefix = f"semantic_{obj.object_id}"

    side_half = [frame_width / 2.0, frame_width / 2.0, half[2]]
    side_half[normal_axis] = depth_half
    for suffix, direction in (("left", -1.0), ("right", 1.0)):
        position = list(surface)
        position[tangent_axis] += direction * (
            half[tangent_axis] + frame_width / 2.0
        )
        _add_visual_box(
            worldbody,
            f"{prefix}_frame_{suffix}",
            position,
            side_half,
            frame_color,
        )

    top_position = list(surface)
    top_position[2] = center[2] + half[2] + frame_width / 2.0
    top_half = [half[tangent_axis] + frame_width, half[tangent_axis] + frame_width, frame_width / 2.0]
    top_half[normal_axis] = depth_half
    _add_visual_box(
        worldbody,
        f"{prefix}_frame_top",
        top_position,
        top_half,
        frame_color,
    )

    panel_position = list(surface)
    panel_position[2] = center[2] + 0.28 * half[2]
    panel_half = [0.54 * half[tangent_axis], 0.54 * half[tangent_axis], 0.13 * half[2]]
    panel_half[normal_axis] = depth_half * 1.2
    _add_visual_box(
        worldbody,
        f"{prefix}_panel_upper",
        panel_position,
        panel_half,
        panel_color,
    )

    handle_position = list(surface)
    handle_position[tangent_axis] += 0.55 * half[tangent_axis]
    handle_position[normal_axis] += direction_to_room * 0.012
    handle_position[2] = center[2] - 0.05 * half[2]
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": f"{prefix}_handle",
            "type": "sphere",
            "pos": _vector(tuple(handle_position)),
            "size": "0.035",
            "rgba": _vector(handle_color),
            "contype": "0",
            "conaffinity": "0",
        },
    )


def _add_visual_box(
    worldbody: ET.Element,
    name: str,
    position: list[float],
    half_extents: list[float],
    rgba: tuple[float, float, float, float],
) -> None:
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": name,
            "type": "box",
            "pos": _vector(tuple(position)),
            "size": _vector(tuple(half_extents)),
            "rgba": _vector(rgba),
            "contype": "0",
            "conaffinity": "0",
        },
    )


def _contrast_color(
    rgba: tuple[float, float, float, float],
    offset: float,
) -> tuple[float, float, float, float]:
    return tuple(
        max(0.02, min(0.98, channel + offset))
        for channel in rgba[:3]
    ) + (rgba[3],)
