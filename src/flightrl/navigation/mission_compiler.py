from __future__ import annotations

import re

from .mission_spec import (
    MissionCommand,
    MissionPlan,
    MissionStep,
    ResolvedMissionPlan,
    ResolvedMissionStep,
    TargetAnchor,
)
from .semantic_scene import Point3, SemanticScene


_MOVE_PREFIX = re.compile(r"^(?:fly|go|navigate|move)\s+to\s+(?:the\s+)?")
_HOLD = re.compile(
    r"^(?:hold|wait)(?:\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*"
    r"(?:s|sec|secs|second|seconds))?$"
)


def compile_mission(text: str, *, default_hold_s: float = 2.0) -> MissionPlan:
    source = text.strip()
    if not source:
        raise ValueError("mission command cannot be empty")
    if default_hold_s <= 0.0:
        raise ValueError("default hold duration must be positive")

    steps = tuple(_compile_clause(clause, default_hold_s) for clause in _clauses(source))
    return MissionPlan(source_text=source, steps=steps)


def resolve_mission(
    plan: MissionPlan,
    scene: SemanticScene,
    *,
    initial_position_m: Point3,
    initial_yaw_rad: float = 0.0,
) -> ResolvedMissionPlan:
    reference = initial_position_m
    yaw = initial_yaw_rad
    previous_index = -1
    resolved: list[ResolvedMissionStep] = []

    for step in plan.steps:
        target_index = -1
        anchor = step.anchor
        if step.command is MissionCommand.GO_TO:
            target = scene.resolve_target(
                step.target_name or "",
                anchor=step.anchor,
                reference_position_m=reference,
            )
            target_index = target.object_index
            anchor = target.anchor
            reference = target.position_m
            yaw = target.yaw_rad
            previous_index = target_index
        elif step.command is MissionCommand.HOLD:
            target_index = previous_index
        elif step.command is MissionCommand.LAND:
            reference = (reference[0], reference[1], scene.room.minimum[2])
        elif step.command is MissionCommand.ABORT:
            pass

        resolved.append(
            ResolvedMissionStep(
                command=step.command,
                target_index=target_index,
                anchor=anchor,
                target_xyz_m=reference,
                target_yaw_rad=yaw,
                duration_s=step.duration_s,
                speed_scale=step.speed_scale,
            )
        )

    return ResolvedMissionPlan(source_text=plan.source_text, steps=tuple(resolved))


def _clauses(text: str) -> tuple[str, ...]:
    normalized = text.lower().strip().rstrip(".!?")
    normalized = re.sub(r"\b(?:and\s+then|then)\b", ",", normalized)
    normalized = re.sub(r"\s+and\s+(?=(?:hold|wait|land|abort)\b)", ",", normalized)
    clauses = tuple(part.strip() for part in normalized.split(",") if part.strip())
    if not clauses:
        raise ValueError("mission command contains no executable clauses")
    return clauses


def _compile_clause(clause: str, default_hold_s: float) -> MissionStep:
    target = _MOVE_PREFIX.sub("", clause, count=1)
    if target != clause:
        anchor = TargetAnchor.PREFERRED
        if target.endswith(" corner"):
            target = target.removesuffix(" corner").strip()
            anchor = TargetAnchor.NEAREST_CORNER
        if not target:
            raise ValueError(f"mission clause {clause!r} has no target")
        return MissionStep(MissionCommand.GO_TO, target_name=target, anchor=anchor)

    hold_match = _HOLD.fullmatch(clause)
    if hold_match:
        duration = float(hold_match.group(1) or default_hold_s)
        return MissionStep(MissionCommand.HOLD, duration_s=duration)
    if clause == "land":
        return MissionStep(MissionCommand.LAND)
    if clause == "abort":
        return MissionStep(MissionCommand.ABORT)
    raise ValueError(
        f"unsupported mission clause {clause!r}; use go/fly/navigate/move to, hold, land, or abort"
    )
