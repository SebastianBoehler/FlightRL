from __future__ import annotations

from math import pi

import numpy as np

from .mission_spec import TargetAnchor
from .room_config import SemanticRoomGenerationConfig
from .room_obstacles import sample_obstacles
from .semantic_scene import (
    Bounds3D,
    Color4,
    RoomAppearance,
    SemanticObject,
    SemanticScene,
)


SEMANTIC_TARGET_CATEGORIES = ("door", "monitor", "sink")
_WALL_SIDES = ("x_min", "x_max", "y_min", "y_max")


def generate_semantic_room(
    seed: int,
    config: SemanticRoomGenerationConfig | None = None,
) -> SemanticScene:
    """Generate a seeded semantic room without binding it to a simulator."""
    settings = config or SemanticRoomGenerationConfig()
    rng = np.random.default_rng(seed)
    width = rng.uniform(*settings.width_range_m)
    depth = rng.uniform(*settings.depth_range_m)
    height = rng.uniform(*settings.height_range_m)
    room = Bounds3D(
        (-0.5 * width, -0.5 * depth, 0.0),
        (0.5 * width, 0.5 * depth, height),
    )
    flight_altitude = float(
        np.clip(
            rng.uniform(*settings.flight_altitude_range_m),
            0.4,
            height - 0.4,
        )
    )
    sides = list(rng.permutation(_WALL_SIDES))
    specifications = _target_specifications(settings, sides, height, rng)
    wall_objects = tuple(
        _wall_object(
            room=room,
            category=category,
            side=side,
            span_m=span,
            depth_m=object_depth,
            z_min=z_min,
            z_max=z_max,
            collision=collision,
            flight_altitude_m=flight_altitude,
            clearance_m=settings.approach_clearance_m,
            rng=rng,
        )
        for category, side, span, object_depth, z_min, z_max, collision in specifications
    )
    obstacles = sample_obstacles(room, wall_objects, settings, rng)
    return SemanticScene(
        room=room,
        objects=(*wall_objects, *obstacles),
        flight_altitude_m=flight_altitude,
        boundary_margin_m=settings.boundary_margin_m,
        appearance=_sample_appearance(settings, rng),
    )


def _wall_object(
    *,
    room: Bounds3D,
    category: str,
    side: str,
    span_m: float,
    depth_m: float,
    z_min: float,
    z_max: float,
    collision: bool,
    flight_altitude_m: float,
    clearance_m: float,
    rng: np.random.Generator,
) -> SemanticObject:
    x_min, y_min, _ = room.minimum
    x_max, y_max, _ = room.maximum
    if side.startswith("x_"):
        tangent = _sample_tangent(rng, y_min, y_max, span_m)
        inward = 1.0 if side == "x_min" else -1.0
        wall = x_min if side == "x_min" else x_max
        inner = wall + inward * depth_m
        bounds = Bounds3D(
            (min(wall, inner), tangent - span_m / 2.0, z_min),
            (max(wall, inner), tangent + span_m / 2.0, z_max),
        )
        approach = (
            inner + inward * clearance_m,
            tangent,
            flight_altitude_m,
        )
        yaw = 0.0 if side == "x_max" else pi
    else:
        tangent = _sample_tangent(rng, x_min, x_max, span_m)
        inward = 1.0 if side == "y_min" else -1.0
        wall = y_min if side == "y_min" else y_max
        inner = wall + inward * depth_m
        bounds = Bounds3D(
            (tangent - span_m / 2.0, min(wall, inner), z_min),
            (tangent + span_m / 2.0, max(wall, inner), z_max),
        )
        approach = (
            tangent,
            inner + inward * clearance_m,
            flight_altitude_m,
        )
        yaw = -pi / 2.0 if side == "y_min" else pi / 2.0
    return SemanticObject(
        object_id=f"{category}_0",
        category=category,
        aliases=_aliases(category),
        bounds=bounds,
        preferred_anchor=TargetAnchor.APPROACH,
        approach_position_m=approach,
        approach_yaw_rad=yaw,
        collision=collision,
        rgba=_randomized_color(category, rng),
    )


def _sample_tangent(
    rng: np.random.Generator,
    minimum: float,
    maximum: float,
    span_m: float,
) -> float:
    margin = span_m / 2.0 + 0.28
    return float(rng.uniform(minimum + margin, maximum - margin))


def _target_specifications(
    config: SemanticRoomGenerationConfig,
    sides: list[str],
    height: float,
    rng: np.random.Generator,
) -> tuple[tuple[str, str, float, float, float, float, bool], ...]:
    scales = rng.uniform(*config.target_scale_range, size=4)
    monitor_center = float(rng.uniform(*config.monitor_center_height_range_m))
    monitor_height = 0.50 * scales[1]
    monitor_min = np.clip(monitor_center - monitor_height / 2.0, 0.30, height - 0.45)
    monitor_max = min(float(monitor_min + monitor_height), height - 0.20)
    return (
        (
            "door",
            sides[0],
            0.86 * scales[0],
            0.04,
            0.0,
            min(2.12 * scales[0] ** 0.25, height - 0.12),
            False,
        ),
        (
            "monitor",
            sides[1],
            0.68 * scales[1],
            0.06,
            float(monitor_min),
            monitor_max,
            False,
        ),
        ("sink", sides[2], 0.78 * scales[2], 0.48, 0.0, 0.92 * scales[2], True),
        ("oven", sides[3], 0.66 * scales[3], 0.50, 0.0, 0.96 * scales[3], True),
    )


def _sample_appearance(
    config: SemanticRoomGenerationConfig,
    rng: np.random.Generator,
) -> RoomAppearance:
    floor = float(rng.uniform(0.18, 0.78))
    wall = float(rng.uniform(0.32, 0.88))
    contrast = float(rng.uniform(*config.appearance_contrast_range))
    floor_tint = rng.uniform(0.88, 1.12, size=3)
    wall_tint = rng.uniform(0.92, 1.08, size=3)
    return RoomAppearance(
        floor_rgb1=_rgb(floor * floor_tint),
        floor_rgb2=_rgb((floor + contrast) * floor_tint),
        wall_rgb1=_rgb(wall * wall_tint),
        wall_rgb2=_rgb((wall + 0.5 * contrast) * wall_tint),
        checker_repeat=float(rng.uniform(*config.checker_repeat_range)),
    )


def _rgb(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.clip(values, 0.02, 0.98))


def _aliases(category: str) -> tuple[str, ...]:
    return {
        "door": ("doorway",),
        "monitor": ("display", "screen"),
        "sink": ("washbasin",),
        "oven": ("stove",),
    }[category]


def _randomized_color(category: str, rng: np.random.Generator) -> Color4:
    base = np.asarray(
        {
            "door": (0.52, 0.36, 0.22),
            "monitor": (0.04, 0.05, 0.06),
            "sink": (0.62, 0.66, 0.68),
            "oven": (0.14, 0.15, 0.16),
        }[category],
        dtype=np.float32,
    )
    color = np.clip(base * rng.uniform(0.75, 1.25), 0.02, 0.95)
    return (float(color[0]), float(color[1]), float(color[2]), 1.0)
