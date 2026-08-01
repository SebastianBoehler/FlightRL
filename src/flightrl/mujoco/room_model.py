from __future__ import annotations

import xml.etree.ElementTree as ET

from flightrl.sixdof.geometry import BoxRoom


def add_box_room_to_mjcf(mjcf: str, room: BoxRoom) -> str:
    root = ET.fromstring(mjcf)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo model has no worldbody")
    resize_room_geometry(
        worldbody,
        (room.x_min, room.y_min, room.z_min),
        (room.x_max, room.y_max, room.z_max),
    )
    for index, obstacle in enumerate(room.obstacles):
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"room_obstacle_{index}",
                "type": "box",
                "pos": _vector(
                    tuple(0.5 * (low + high) for low, high in obstacle.bounds)
                ),
                "size": _vector(
                    tuple(0.5 * (high - low) for low, high in obstacle.bounds)
                ),
                "rgba": "0.35 0.38 0.42 1",
                "contype": "1",
                "conaffinity": "1",
            },
        )
    return ET.tostring(root, encoding="unicode")


def resize_room_geometry(
    worldbody: ET.Element,
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> None:
    center = tuple(0.5 * (low + high) for low, high in zip(minimum, maximum))
    half = tuple(0.5 * (high - low) for low, high in zip(minimum, maximum))
    x_min, y_min, z_min = minimum
    x_max, y_max, z_max = maximum
    center_x, center_y, center_z = center
    half_x, half_y, half_z = half
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


def _vector(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:g}" for value in values)
