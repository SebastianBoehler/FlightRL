"""Low-rate typed mission primitives compiled outside the control loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite
import re


MISSION_PROGRAM_VERSION = 1
MISSION_PRIMITIVE_FIELDS = (
    "kind",
    "target_index",
    "max_speed_m_s",
    "minimum_altitude_m",
    "maximum_altitude_m",
    "timeout_s",
    "standoff_m",
)
_TARGET_NAME = re.compile(r"[a-z0-9][a-z0-9_]*")


class MissionPrimitiveKind(IntEnum):
    SEARCH = 1
    APPROACH = 2
    INSPECT = 3
    TRACK = 4
    HOLD = 5
    RETURN = 6
    LAND = 7
    ABORT = 8


TARGETED_PRIMITIVES = frozenset(
    {
        MissionPrimitiveKind.SEARCH,
        MissionPrimitiveKind.APPROACH,
        MissionPrimitiveKind.INSPECT,
        MissionPrimitiveKind.TRACK,
    }
)


@dataclass(frozen=True, slots=True)
class MissionConstraints:
    max_speed_m_s: float = 1.0
    minimum_altitude_m: float = 0.0
    maximum_altitude_m: float = 10.0
    timeout_s: float = 30.0
    standoff_m: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.max_speed_m_s,
            self.minimum_altitude_m,
            self.maximum_altitude_m,
            self.timeout_s,
            self.standoff_m,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("mission constraints must be finite")
        if self.max_speed_m_s <= 0.0 or self.timeout_s <= 0.0:
            raise ValueError("mission speed and timeout must be positive")
        if self.minimum_altitude_m < 0.0:
            raise ValueError("mission minimum altitude must be non-negative")
        if self.maximum_altitude_m <= self.minimum_altitude_m:
            raise ValueError("mission maximum altitude must exceed minimum altitude")
        if self.standoff_m < 0.0:
            raise ValueError("mission standoff must be non-negative")


@dataclass(frozen=True, slots=True)
class MissionPrimitive:
    kind: MissionPrimitiveKind
    target_name: str | None = None
    constraints: MissionConstraints = MissionConstraints()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MissionPrimitiveKind):
            raise TypeError("mission primitive kind must be a MissionPrimitiveKind")
        if not isinstance(self.constraints, MissionConstraints):
            raise TypeError("mission primitive constraints must be MissionConstraints")
        if self.kind in TARGETED_PRIMITIVES:
            if self.target_name is None:
                raise ValueError(f"{self.kind.name.lower()} requires a target")
            _require_target_name(self.target_name)
        elif self.target_name is not None:
            raise ValueError(f"{self.kind.name.lower()} cannot name a target")

    def to_row(self, target_index: int) -> tuple[float, ...]:
        limits = self.constraints
        return (
            float(self.kind),
            float(target_index),
            limits.max_speed_m_s,
            limits.minimum_altitude_m,
            limits.maximum_altitude_m,
            limits.timeout_s,
            limits.standoff_m,
        )


@dataclass(frozen=True, slots=True)
class MissionProgram:
    source_text: str
    primitives: tuple[MissionPrimitive, ...]
    target_vocabulary: tuple[str, ...]
    contract_version: int = MISSION_PROGRAM_VERSION

    def __post_init__(self) -> None:
        if not self.source_text.strip():
            raise ValueError("mission source text cannot be empty")
        if not self.primitives or not all(
            isinstance(primitive, MissionPrimitive) for primitive in self.primitives
        ):
            raise ValueError("mission program requires typed primitives")
        if self.contract_version != MISSION_PROGRAM_VERSION:
            raise ValueError(f"unsupported mission program version {self.contract_version}")
        for target in self.target_vocabulary:
            _require_target_name(target)
        if len(set(self.target_vocabulary)) != len(self.target_vocabulary):
            raise ValueError("mission target vocabulary must be unique")
        vocabulary = set(self.target_vocabulary)
        for primitive in self.primitives:
            if primitive.target_name is not None and primitive.target_name not in vocabulary:
                raise ValueError(
                    f"mission target {primitive.target_name!r} is not in the target vocabulary"
                )

    def to_rows(self) -> tuple[tuple[float, ...], ...]:
        indices = {
            target: index for index, target in enumerate(self.target_vocabulary)
        }
        return tuple(
            primitive.to_row(
                -1 if primitive.target_name is None else indices[primitive.target_name]
            )
            for primitive in self.primitives
        )


def _require_target_name(value: str) -> None:
    if _TARGET_NAME.fullmatch(value) is None:
        raise ValueError("mission target names must be lowercase snake_case identifiers")
