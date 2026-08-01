from __future__ import annotations

import numpy as np

from .mission_spec import TargetAnchor
from .room_config import SemanticRoomGenerationConfig
from .semantic_scene import Bounds3D, SemanticObject


def sample_obstacles(
    room: Bounds3D,
    existing: tuple[SemanticObject, ...],
    config: SemanticRoomGenerationConfig,
    rng: np.random.Generator,
) -> tuple[SemanticObject, ...]:
    low, high = config.obstacle_count_range
    count = int(rng.integers(low, high + 1))
    obstacles: list[SemanticObject] = []
    for index in range(count):
        for _attempt in range(100):
            profile = str(rng.choice(config.obstacle_profiles))
            width, depth, height, shape = _dimensions(profile, rng)
            x = float(
                rng.uniform(
                    room.minimum[0] + width / 2.0 + 0.35,
                    room.maximum[0] - width / 2.0 - 0.35,
                )
            )
            y = float(
                rng.uniform(
                    room.minimum[1] + depth / 2.0 + 0.35,
                    room.maximum[1] - depth / 2.0 - 0.35,
                )
            )
            bounds = Bounds3D(
                (x - width / 2.0, y - depth / 2.0, 0.0),
                (x + width / 2.0, y + depth / 2.0, min(height, room.maximum[2] - 0.1)),
            )
            occupied = (*existing, *obstacles)
            if (
                np.hypot(x, y) < 1.05
                or any(_xy_overlaps(bounds, obj.bounds, margin=0.30) for obj in occupied)
                or any(
                    obj.approach_position_m is not None
                    and _contains_xy(bounds, obj.approach_position_m, margin=0.40)
                    for obj in existing
                )
            ):
                continue
            base = rng.uniform(0.12, 0.82, size=3)
            obstacles.append(
                SemanticObject(
                    object_id=f"obstacle_{index}",
                    category="obstacle",
                    bounds=bounds,
                    preferred_anchor=TargetAnchor.CENTER,
                    collision=True,
                    rgba=(*map(float, base), 1.0),
                    shape=shape,
                )
            )
            break
    return tuple(obstacles)


def _dimensions(
    profile: str,
    rng: np.random.Generator,
) -> tuple[float, float, float, str]:
    if profile == "crate":
        return (
            float(rng.uniform(0.30, 0.85)),
            float(rng.uniform(0.30, 0.85)),
            float(rng.uniform(0.30, 0.90)),
            "box",
        )
    if profile == "cabinet":
        return (
            float(rng.uniform(0.40, 0.90)),
            float(rng.uniform(0.28, 0.60)),
            float(rng.uniform(1.00, 2.00)),
            "box",
        )
    if profile == "partition":
        return (
            float(rng.uniform(0.90, 1.60)),
            float(rng.uniform(0.12, 0.24)),
            float(rng.uniform(1.10, 2.10)),
            "box",
        )
    radius = float(rng.uniform(0.16, 0.38))
    return (2.0 * radius, 2.0 * radius, float(rng.uniform(1.2, 2.3)), "cylinder")


def _xy_overlaps(first: Bounds3D, second: Bounds3D, *, margin: float) -> bool:
    return not (
        first.maximum[0] + margin < second.minimum[0]
        or first.minimum[0] - margin > second.maximum[0]
        or first.maximum[1] + margin < second.minimum[1]
        or first.minimum[1] - margin > second.maximum[1]
    )


def _contains_xy(
    bounds: Bounds3D,
    point: tuple[float, float, float],
    *,
    margin: float,
) -> bool:
    return (
        bounds.minimum[0] - margin <= point[0] <= bounds.maximum[0] + margin
        and bounds.minimum[1] - margin <= point[1] <= bounds.maximum[1] + margin
    )
