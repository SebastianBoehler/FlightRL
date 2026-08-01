from __future__ import annotations

import re
from dataclasses import dataclass
from math import atan2, isfinite

from .mission_spec import TargetAnchor


Point3 = tuple[float, float, float]
Color3 = tuple[float, float, float]
Color4 = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Bounds3D:
    minimum: Point3
    maximum: Point3

    def __post_init__(self) -> None:
        values = (*self.minimum, *self.maximum)
        if not all(isfinite(value) for value in values):
            raise ValueError("scene bounds must be finite")
        if any(low >= high for low, high in zip(self.minimum, self.maximum, strict=True)):
            raise ValueError("scene bounds minimum must be below maximum on every axis")

    @property
    def center(self) -> Point3:
        return tuple(
            0.5 * (low + high)
            for low, high in zip(self.minimum, self.maximum, strict=True)
        )

    @property
    def half_extents(self) -> Point3:
        return tuple(
            0.5 * (high - low)
            for low, high in zip(self.minimum, self.maximum, strict=True)
        )

    def contains(self, point: Point3, margin: float = 0.0) -> bool:
        return all(
            low + margin <= value <= high - margin
            for value, low, high in zip(point, self.minimum, self.maximum, strict=True)
        )


@dataclass(frozen=True, slots=True)
class SemanticObject:
    object_id: str
    category: str
    bounds: Bounds3D
    aliases: tuple[str, ...] = ()
    preferred_anchor: TargetAnchor = TargetAnchor.APPROACH
    approach_position_m: Point3 | None = None
    approach_yaw_rad: float | None = None
    collision: bool = True
    rgba: Color4 = (0.45, 0.45, 0.45, 1.0)
    shape: str = "box"

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.object_id) is None:
            raise ValueError("semantic object id must use lowercase letters, digits, '-' or '_'")
        if not _normalize_name(self.category):
            raise ValueError("semantic object category cannot be empty")
        if self.preferred_anchor is TargetAnchor.APPROACH and self.approach_position_m is None:
            raise ValueError("objects with an approach anchor require an approach position")
        if self.approach_position_m is not None and not all(
            isfinite(value) for value in self.approach_position_m
        ):
            raise ValueError("approach position must be finite")
        if self.approach_yaw_rad is not None and not isfinite(self.approach_yaw_rad):
            raise ValueError("approach yaw must be finite")
        if len(self.rgba) != 4 or not all(0.0 <= value <= 1.0 for value in self.rgba):
            raise ValueError("object RGBA values must be in [0, 1]")
        if self.shape not in {"box", "cylinder"}:
            raise ValueError("semantic object shape must be box or cylinder")

    @property
    def names(self) -> tuple[str, ...]:
        return (self.object_id, self.category, *self.aliases)


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    object_index: int
    object_id: str
    anchor: TargetAnchor
    position_m: Point3
    yaw_rad: float


@dataclass(frozen=True, slots=True)
class RoomAppearance:
    floor_rgb1: Color3 = (0.58, 0.60, 0.62)
    floor_rgb2: Color3 = (0.72, 0.74, 0.76)
    wall_rgb1: Color3 = (0.72, 0.74, 0.76)
    wall_rgb2: Color3 = (0.86, 0.88, 0.90)
    checker_repeat: float = 4.0

    def __post_init__(self) -> None:
        colors = (*self.floor_rgb1, *self.floor_rgb2, *self.wall_rgb1, *self.wall_rgb2)
        if not all(0.0 <= value <= 1.0 for value in colors):
            raise ValueError("room appearance RGB values must be in [0, 1]")
        if self.checker_repeat <= 0.0 or not isfinite(self.checker_repeat):
            raise ValueError("room checker repeat must be finite and positive")


@dataclass(frozen=True, slots=True)
class SemanticScene:
    room: Bounds3D
    objects: tuple[SemanticObject, ...]
    flight_altitude_m: float = 0.8
    waypoint_clearance_m: float = 0.25
    boundary_margin_m: float = 0.12
    appearance: RoomAppearance = RoomAppearance()

    def __post_init__(self) -> None:
        if not self.room.minimum[2] < self.flight_altitude_m < self.room.maximum[2]:
            raise ValueError("flight altitude must be inside the room")
        aliases: dict[str, str] = {}
        object_ids: set[str] = set()
        for obj in self.objects:
            if obj.object_id in object_ids:
                raise ValueError(f"duplicate semantic object id {obj.object_id!r}")
            object_ids.add(obj.object_id)
            if not (
                self.room.contains(obj.bounds.minimum)
                and self.room.contains(obj.bounds.maximum)
            ):
                raise ValueError(f"object {obj.object_id!r} is outside the room")
            for name in (obj.object_id, *obj.aliases):
                normalized = _normalize_name(name)
                if not normalized:
                    raise ValueError(f"semantic object {obj.object_id!r} has an empty alias")
                previous = aliases.get(normalized)
                if previous is not None and previous != obj.object_id:
                    raise ValueError(
                        f"semantic name {name!r} is shared by {previous!r} and {obj.object_id!r}"
                    )
                aliases[normalized] = obj.object_id

    def object_by_name(self, name: str) -> tuple[int, SemanticObject]:
        normalized = _normalize_name(name)
        matches = [
            (index, obj)
            for index, obj in enumerate(self.objects)
            if normalized in {_normalize_name(candidate) for candidate in obj.names}
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            ids = ", ".join(obj.object_id for _, obj in matches)
            raise KeyError(f"ambiguous semantic target {name!r}; use one of: {ids}")
        known = ", ".join(obj.object_id for obj in self.objects)
        raise KeyError(f"unknown semantic target {name!r}; known targets: {known}")

    def resolve_target(
        self,
        name: str,
        *,
        anchor: TargetAnchor,
        reference_position_m: Point3,
    ) -> ResolvedTarget:
        object_index, obj = self.object_by_name(name)
        selected_anchor = obj.preferred_anchor if anchor is TargetAnchor.PREFERRED else anchor
        if selected_anchor is TargetAnchor.APPROACH:
            if obj.approach_position_m is None:
                raise ValueError(f"target {obj.object_id!r} has no approach point")
            position = obj.approach_position_m
        elif selected_anchor is TargetAnchor.NEAREST_CORNER:
            position = self._nearest_safe_corner(obj, reference_position_m)
        elif selected_anchor is TargetAnchor.CENTER:
            if obj.collision:
                raise ValueError(f"cannot target the center of collidable object {obj.object_id!r}")
            position = (obj.bounds.center[0], obj.bounds.center[1], self.flight_altitude_m)
        else:
            raise ValueError(f"unsupported target anchor {selected_anchor!r}")

        if not self.room.contains(position, margin=self.boundary_margin_m):
            raise ValueError(f"resolved target for {obj.object_id!r} is outside the safe room bounds")
        yaw = obj.approach_yaw_rad
        if yaw is None:
            yaw = atan2(obj.bounds.center[1] - position[1], obj.bounds.center[0] - position[0])
        return ResolvedTarget(object_index, obj.object_id, selected_anchor, position, yaw)

    def _nearest_safe_corner(
        self,
        obj: SemanticObject,
        reference_position_m: Point3,
    ) -> Point3:
        low_x, low_y, _ = obj.bounds.minimum
        high_x, high_y, _ = obj.bounds.maximum
        clearance = self.waypoint_clearance_m
        candidates = (
            (low_x - clearance, low_y - clearance, self.flight_altitude_m),
            (low_x - clearance, high_y + clearance, self.flight_altitude_m),
            (high_x + clearance, low_y - clearance, self.flight_altitude_m),
            (high_x + clearance, high_y + clearance, self.flight_altitude_m),
        )
        safe = [
            point
            for point in candidates
            if self.room.contains(point, margin=self.boundary_margin_m)
        ]
        if not safe:
            raise ValueError(f"target {obj.object_id!r} has no safe corner waypoint")
        return min(
            safe,
            key=lambda point: (point[0] - reference_position_m[0]) ** 2
            + (point[1] - reference_position_m[1]) ** 2,
        )


def _normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())
