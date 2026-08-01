from __future__ import annotations

from typing import Any, Mapping

from flightrl.puffer4_door_challenges import (
    DoorCameraLatencyTransform,
    DoorPixelNoiseTransform,
)
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)


PIXEL_NOISE_CHALLENGE_SEED = 20_260_735
DOOR_CHALLENGE_NAMES = (
    "fixed-dark",
    "obstacle-present",
    "room-footprint-1p2x",
    "pixel-noise",
    "camera-latency-92ms",
)
_EXPECTED_BASELINE = {
    "fixed-dark": {
        "camera_mean_min": 18.0,
        "camera_mean_max": 110.0,
        "camera_randomization": 0.0,
    },
    "obstacle-present": {"obstacle_probability": 0.0},
    "room-footprint-1p2x": {
        "room_x_min": -2.0,
        "room_x_max": 2.0,
        "room_y_min": -2.0,
        "room_y_max": 2.0,
    },
    "pixel-noise": {},
    "camera-latency-92ms": {},
}
_ENV_OVERRIDES = {
    "fixed-dark": {
        "camera_mean_min": 20.0,
        "camera_mean_max": 20.0,
    },
    "obstacle-present": {"obstacle_probability": 1.0},
    "room-footprint-1p2x": {
        "room_x_min": -2.4,
        "room_x_max": 2.4,
        "room_y_min": -2.4,
        "room_y_max": 2.4,
    },
    "pixel-noise": {},
    "camera-latency-92ms": {},
}
_VARIABLES = {
    "fixed-dark": "fixed target exposure",
    "obstacle-present": "single route-intersecting obstacle probability",
    "room-footprint-1p2x": "horizontal room footprint scale",
    "pixel-noise": "actor-input pixel noise",
    "camera-latency-92ms": "additional fixed camera latency",
}
_LIMITATIONS = {
    "fixed-dark": (
        "The native low-light marginal flag remains category zero because "
        "composite camera randomization stays disabled."
    ),
    "obstacle-present": (
        "The grammar contains one axis-aligned route-intersecting cuboid, "
        "not general clutter or multi-obstacle planning."
    ),
    "room-footprint-1p2x": (
        "This changes one rectangular room footprint; it does not introduce "
        "a disjoint room topology and retains the baseline horizon."
    ),
    "pixel-noise": (
        "Detector phase/evidence remain clean, so this isolates actor raster "
        "robustness rather than end-to-end detector robustness."
    ),
    "camera-latency-92ms": (
        "The fixed delay is additional to detector sample-and-hold and does "
        "not model latency jitter, drops, or asynchronous transport."
    ),
}


def resolve_door_challenge(
    name: str,
    baseline_env: Mapping[str, Any],
    *,
    agent_count: int,
) -> tuple[dict[str, Any], object | None, dict[str, Any]]:
    if name not in DOOR_CHALLENGE_NAMES:
        raise ValueError(
            f"unknown fixed-door challenge {name!r}; "
            f"expected one of {DOOR_CHALLENGE_NAMES}"
        )
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT.verify_env(baseline_env)
    expected = _EXPECTED_BASELINE[name]
    mismatched = {
        key: (baseline_env.get(key), value)
        for key, value in expected.items()
        if baseline_env.get(key) != value
    }
    if mismatched:
        raise ValueError(
            f"fixed-door challenge baseline does not match: {mismatched}"
        )
    overrides = dict(_ENV_OVERRIDES[name])
    resolved = dict(baseline_env)
    resolved.update(overrides)
    transform = _observation_transform(name, agent_count)
    mechanism = (
        None if transform is None else transform.mechanism_report()
    )
    return resolved, transform, {
        "schema_version": 1,
        "name": name,
        "single_controlled_variable": _VARIABLES[name],
        "environment_overrides": overrides,
        "observation_transform": mechanism,
        "matched_control_required": True,
        "combined_with_other_challenges": False,
        "limitation": _LIMITATIONS[name],
    }


def _observation_transform(name: str, agent_count: int) -> object | None:
    if name == "pixel-noise":
        return DoorPixelNoiseTransform(
            agent_count=agent_count,
            seed=PIXEL_NOISE_CHALLENGE_SEED,
        )
    if name == "camera-latency-92ms":
        runtime = FIXED_DOOR_EVIDENCE_AGE_CONTRACT
        return DoorCameraLatencyTransform(
            agent_count=agent_count,
            delay_steps=6,
            control_dt_s=runtime.control_dt_s,
            maximum_evidence_age_s=runtime.maximum_evidence_age_s,
        )
    return None
