from __future__ import annotations

import pytest

from flightrl.puffer4_door_challenge_specs import (
    DOOR_CHALLENGE_NAMES,
    resolve_door_challenge,
)
from flightrl.puffer4_door_challenges import (
    DoorCameraLatencyTransform,
    DoorPixelNoiseTransform,
)


def _baseline() -> dict:
    return {
        "control_dt": 1.0 / 65.0,
        "maximum_evidence_age_s": 1.0,
        "camera_mean_min": 18.0,
        "camera_mean_max": 110.0,
        "camera_randomization": 0.0,
        "obstacle_probability": 0.0,
        "layout_diversity": 1.0,
        "room_x_min": -2.0,
        "room_x_max": 2.0,
        "room_y_min": -2.0,
        "room_y_max": 2.0,
    }


@pytest.mark.parametrize(
    ("name", "changed"),
    (
        ("fixed-dark", {"camera_mean_min", "camera_mean_max"}),
        ("obstacle-present", {"obstacle_probability"}),
        (
            "room-footprint-1p2x",
            {"room_x_min", "room_x_max", "room_y_min", "room_y_max"},
        ),
        ("pixel-noise", set()),
        ("camera-latency-92ms", set()),
    ),
)
def test_challenges_change_only_the_registered_variable(
    name: str,
    changed: set[str],
) -> None:
    baseline = _baseline()
    resolved, transform, report = resolve_door_challenge(
        name,
        baseline,
        agent_count=4,
    )

    assert {
        key for key in baseline if resolved[key] != baseline[key]
    } == changed
    assert report["name"] == name
    assert report["single_controlled_variable"]
    assert report["environment_overrides"] == {
        key: resolved[key] for key in changed
    }
    if name == "pixel-noise":
        assert isinstance(transform, DoorPixelNoiseTransform)
    elif name == "camera-latency-92ms":
        assert isinstance(transform, DoorCameraLatencyTransform)
    else:
        assert transform is None


def test_fixed_dark_does_not_enable_composite_camera_randomization() -> None:
    resolved, _, _ = resolve_door_challenge(
        "fixed-dark",
        _baseline(),
        agent_count=4,
    )

    assert resolved["camera_mean_min"] == 20.0
    assert resolved["camera_mean_max"] == 20.0
    assert resolved["camera_randomization"] == 0.0


def test_challenge_rejects_wrong_baseline_and_unknown_combinations() -> None:
    with pytest.raises(ValueError, match="baseline"):
        resolve_door_challenge(
            "obstacle-present",
            _baseline() | {"obstacle_probability": 0.5},
            agent_count=4,
        )
    with pytest.raises(ValueError, match="unknown"):
        resolve_door_challenge(
            "dark-with-obstacles",
            _baseline(),
            agent_count=4,
        )

    assert "dark-with-obstacles" not in DOOR_CHALLENGE_NAMES
