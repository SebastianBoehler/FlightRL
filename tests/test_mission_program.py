from __future__ import annotations

import pytest

from flightrl.navigation import (
    MISSION_PRIMITIVE_FIELDS,
    MissionConstraints,
    MissionPrimitive,
    MissionPrimitiveKind,
    MissionProgram,
)


def test_composes_general_mission_from_fixed_low_rate_primitives() -> None:
    constraints = MissionConstraints(
        max_speed_m_s=3.0,
        minimum_altitude_m=1.0,
        maximum_altitude_m=20.0,
        timeout_s=45.0,
        standoff_m=2.0,
    )
    program = MissionProgram(
        source_text="Find the red car, inspect it, return, and land",
        target_vocabulary=("red_car",),
        primitives=(
            MissionPrimitive(MissionPrimitiveKind.SEARCH, "red_car", constraints),
            MissionPrimitive(MissionPrimitiveKind.APPROACH, "red_car", constraints),
            MissionPrimitive(MissionPrimitiveKind.INSPECT, "red_car", constraints),
            MissionPrimitive(MissionPrimitiveKind.RETURN),
            MissionPrimitive(MissionPrimitiveKind.LAND),
        ),
    )

    rows = program.to_rows()

    assert len(rows) == 5
    assert all(len(row) == len(MISSION_PRIMITIVE_FIELDS) for row in rows)
    assert rows[0][0] == float(MissionPrimitiveKind.SEARCH)
    assert rows[0][1] == 0.0
    assert rows[3][1] == -1.0


@pytest.mark.parametrize(
    "kind",
    (
        MissionPrimitiveKind.SEARCH,
        MissionPrimitiveKind.APPROACH,
        MissionPrimitiveKind.INSPECT,
        MissionPrimitiveKind.TRACK,
    ),
)
def test_targeted_primitives_require_explicit_target(kind) -> None:
    with pytest.raises(ValueError, match="requires a target"):
        MissionPrimitive(kind)


def test_control_primitives_cannot_smuggle_target_text_into_fast_loop() -> None:
    with pytest.raises(ValueError, match="cannot name a target"):
        MissionPrimitive(MissionPrimitiveKind.ABORT, "red_car")


def test_program_rejects_unbound_or_ambiguous_target_vocabulary() -> None:
    primitive = MissionPrimitive(MissionPrimitiveKind.SEARCH, "red_car")
    with pytest.raises(ValueError, match="not in the target vocabulary"):
        MissionProgram("search", (primitive,), ())
    with pytest.raises(ValueError, match="unique"):
        MissionProgram("search", (primitive,), ("red_car", "red_car"))


def test_constraints_are_finite_and_physically_ordered() -> None:
    with pytest.raises(ValueError, match="altitude"):
        MissionConstraints(minimum_altitude_m=5.0, maximum_altitude_m=2.0)
    with pytest.raises(ValueError, match="positive"):
        MissionConstraints(max_speed_m_s=0.0)
