from __future__ import annotations

import pytest

from flightrl.navigation import (
    MISSION_STEP_FIELDS,
    Bounds3D,
    MissionCommand,
    SemanticObject,
    SemanticScene,
    TargetAnchor,
    compile_mission,
    resolve_mission,
)


def _scene() -> SemanticScene:
    return SemanticScene(
        room=Bounds3D((-2.0, -2.0, 0.0), (2.0, 2.0, 2.5)),
        objects=(
            SemanticObject(
                object_id="desk",
                category="table",
                aliases=("work table",),
                bounds=Bounds3D((-0.5, 0.4, 0.0), (0.5, 1.1, 0.75)),
                preferred_anchor=TargetAnchor.APPROACH,
                approach_position_m=(0.0, 0.05, 0.8),
                approach_yaw_rad=1.57,
                rgba=(0.35, 0.20, 0.10, 1.0),
            ),
            SemanticObject(
                object_id="door",
                category="doorway",
                aliases=("exit",),
                bounds=Bounds3D((1.94, -0.45, 0.0), (1.99, 0.45, 2.1)),
                preferred_anchor=TargetAnchor.APPROACH,
                approach_position_m=(1.45, 0.0, 0.8),
                approach_yaw_rad=0.0,
                collision=False,
                rgba=(0.15, 0.25, 0.35, 1.0),
            ),
        ),
    )


def test_compiles_and_resolves_bounded_semantic_mission() -> None:
    plan = compile_mission(
        "Fly to the desk corner, hold for 3 seconds, then go to the door"
    )

    assert [step.command for step in plan.steps] == [
        MissionCommand.GO_TO,
        MissionCommand.HOLD,
        MissionCommand.GO_TO,
    ]
    assert plan.steps[0].target_name == "desk"
    assert plan.steps[0].anchor is TargetAnchor.NEAREST_CORNER
    assert plan.steps[1].duration_s == 3.0

    resolved = resolve_mission(plan, _scene(), initial_position_m=(-1.5, -1.2, 0.8))

    assert resolved.steps[0].target_xyz_m == pytest.approx((-0.75, 0.15, 0.8))
    assert resolved.steps[1].target_xyz_m == resolved.steps[0].target_xyz_m
    assert resolved.steps[2].target_xyz_m == (1.45, 0.0, 0.8)
    assert all(len(row) == len(MISSION_STEP_FIELDS) for row in resolved.to_rows())


def test_object_alias_resolves_to_stable_scene_index() -> None:
    plan = compile_mission("go to the exit and hold")
    resolved = resolve_mission(plan, _scene(), initial_position_m=(0.0, 0.0, 0.8))

    assert resolved.steps[0].target_index == 1
    assert resolved.steps[0].anchor is TargetAnchor.APPROACH
    assert resolved.steps[1].target_index == 1


def test_unknown_language_and_unknown_targets_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="unsupported mission clause"):
        compile_mission("explore the whole room")

    plan = compile_mission("go to the sofa")
    with pytest.raises(KeyError, match="unknown semantic target"):
        resolve_mission(plan, _scene(), initial_position_m=(0.0, 0.0, 0.8))


def test_duplicate_semantic_aliases_are_rejected() -> None:
    bounds = Bounds3D((-0.4, -0.4, 0.0), (0.4, 0.4, 0.8))
    with pytest.raises(ValueError, match="shared"):
        SemanticScene(
            room=Bounds3D((-2.0, -2.0, 0.0), (2.0, 2.0, 2.5)),
            objects=(
                SemanticObject(
                    "desk",
                    "table",
                    bounds,
                    aliases=("target",),
                    approach_position_m=(0.0, -0.8, 0.8),
                ),
                SemanticObject(
                    "door",
                    "doorway",
                    bounds,
                    aliases=("target",),
                    approach_position_m=(0.8, 0.0, 0.8),
                ),
            ),
        )


def test_repeated_categories_require_an_object_id() -> None:
    scene = SemanticScene(
        room=Bounds3D((-2.0, -2.0, 0.0), (2.0, 2.0, 2.5)),
        objects=(
            SemanticObject(
                "desk_left",
                "desk",
                Bounds3D((-1.2, 0.2, 0.0), (-0.4, 1.0, 0.8)),
                approach_position_m=(-0.8, -0.1, 0.8),
            ),
            SemanticObject(
                "desk_right",
                "desk",
                Bounds3D((0.4, 0.2, 0.0), (1.2, 1.0, 0.8)),
                approach_position_m=(0.8, -0.1, 0.8),
            ),
        ),
    )

    with pytest.raises(KeyError, match="ambiguous semantic target"):
        scene.object_by_name("desk")
    assert scene.object_by_name("desk left")[1].object_id == "desk_left"


def test_duplicate_object_ids_are_rejected() -> None:
    bounds = Bounds3D((-0.4, -0.4, 0.0), (0.4, 0.4, 0.8))
    objects = (
        SemanticObject("desk", "table", bounds, approach_position_m=(0.0, -0.8, 0.8)),
        SemanticObject("desk", "table", bounds, approach_position_m=(0.8, 0.0, 0.8)),
    )

    with pytest.raises(ValueError, match="duplicate semantic object id"):
        SemanticScene(
            room=Bounds3D((-2.0, -2.0, 0.0), (2.0, 2.0, 2.5)),
            objects=objects,
        )
