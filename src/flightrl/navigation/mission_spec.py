from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite


MISSION_CONTRACT_VERSION = 1
MISSION_STEP_FIELDS = (
    "command",
    "target_index",
    "anchor",
    "target_x_m",
    "target_y_m",
    "target_z_m",
    "target_yaw_rad",
    "duration_s",
    "speed_scale",
)


class MissionCommand(IntEnum):
    GO_TO = 1
    HOLD = 2
    LAND = 3
    ABORT = 4


class TargetAnchor(IntEnum):
    PREFERRED = 0
    CENTER = 1
    NEAREST_CORNER = 2
    APPROACH = 3


@dataclass(frozen=True, slots=True)
class MissionStep:
    command: MissionCommand
    target_name: str | None = None
    anchor: TargetAnchor = TargetAnchor.PREFERRED
    duration_s: float = 0.0
    speed_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.command is MissionCommand.GO_TO and not self.target_name:
            raise ValueError("go-to mission steps require a target name")
        if self.command is not MissionCommand.GO_TO and self.target_name is not None:
            raise ValueError(f"{self.command.name.lower()} mission steps cannot name a target")
        if not isfinite(self.duration_s) or self.duration_s < 0.0:
            raise ValueError("mission duration must be finite and non-negative")
        if not isfinite(self.speed_scale) or not 0.0 < self.speed_scale <= 1.0:
            raise ValueError("mission speed scale must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class MissionPlan:
    source_text: str
    steps: tuple[MissionStep, ...]
    contract_version: int = MISSION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.source_text.strip():
            raise ValueError("mission source text cannot be empty")
        if not self.steps:
            raise ValueError("mission plan must contain at least one step")
        if self.contract_version != MISSION_CONTRACT_VERSION:
            raise ValueError(f"unsupported mission contract version {self.contract_version}")


@dataclass(frozen=True, slots=True)
class ResolvedMissionStep:
    command: MissionCommand
    target_index: int
    anchor: TargetAnchor
    target_xyz_m: tuple[float, float, float]
    target_yaw_rad: float
    duration_s: float
    speed_scale: float

    def to_row(self) -> tuple[float, ...]:
        return (
            float(self.command),
            float(self.target_index),
            float(self.anchor),
            *self.target_xyz_m,
            self.target_yaw_rad,
            self.duration_s,
            self.speed_scale,
        )


@dataclass(frozen=True, slots=True)
class ResolvedMissionPlan:
    source_text: str
    steps: tuple[ResolvedMissionStep, ...]
    contract_version: int = MISSION_CONTRACT_VERSION

    def to_rows(self) -> tuple[tuple[float, ...], ...]:
        return tuple(step.to_row() for step in self.steps)
