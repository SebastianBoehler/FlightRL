from __future__ import annotations

from math import pi

import numpy as np
import pytest

from flightrl.puffer4_door_observation import (
    DOOR_EVIDENCE_DIM,
    DOOR_PHASE_DIM,
    DOOR_PROPRIO_DIM,
    build_door_proprioception,
    door_observation_origin,
)


def _telemetry(**overrides: float) -> dict[str, float]:
    values = {
        "stateEstimate.x": 0.0,
        "stateEstimate.y": 0.0,
        "stateEstimate.z": 1.5,
        "stateEstimate.vx": 1.0,
        "stateEstimate.vy": 0.0,
        "stateEstimate.vz": 0.5,
        "stateEstimate.roll": 0.0,
        "stateEstimate.pitch": 0.0,
        "stateEstimate.yaw": 0.0,
        "gyro.x": 180.0 / pi,
        "gyro.y": 360.0 / pi,
        "gyro.z": 180.0 / pi,
    }
    values.update(overrides)
    return values


def test_live_proprioception_matches_level_native_scaling() -> None:
    telemetry = _telemetry()
    origin = door_observation_origin(telemetry)

    values = build_door_proprioception(
        telemetry,
        origin,
        np.asarray((0.25, -0.5), dtype=np.float32),
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        np.asarray((0.9, -0.25, 0.5, 0.4, 0.2), dtype=np.float32),
    )

    assert values.shape == (DOOR_PROPRIO_DIM,)
    assert values == pytest.approx(
        (
            1.0,
            0.0,
            1.0,
            1.0 / 6.0,
            2.0 / 6.0,
            0.25,
            0.0,
            0.0,
            1.0,
            0.6,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.25,
            -0.5,
            1.0,
            0.0,
            0.0,
            0.0,
            0.9,
            -0.25,
            0.5,
            0.4,
            0.2,
        ),
        abs=1.0e-6,
    )


def test_live_proprioception_uses_takeoff_relative_frame() -> None:
    initial = _telemetry(
        **{
            "stateEstimate.vx": 0.0,
            "stateEstimate.vz": 0.0,
        }
    )
    current = _telemetry(
        **{
            "stateEstimate.x": 1.0,
            "stateEstimate.y": 2.0,
            "stateEstimate.vx": 0.0,
            "stateEstimate.vy": 1.0,
            "stateEstimate.vz": 0.0,
            "stateEstimate.yaw": 90.0,
        }
    )

    values = build_door_proprioception(
        current,
        door_observation_origin(initial),
        np.zeros(2, dtype=np.float32),
        np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float32),
        np.zeros(DOOR_EVIDENCE_DIM, dtype=np.float32),
    )

    assert values[0] == pytest.approx(1.0)
    assert values[10:13] == pytest.approx((0.25, 0.5, 0.0))
    assert values[13:15] == pytest.approx((1.0, 0.0), abs=1.0e-6)


def test_live_proprioception_rejects_missing_telemetry() -> None:
    telemetry = _telemetry()
    del telemetry["gyro.z"]

    with pytest.raises(KeyError, match="gyro.z"):
        build_door_proprioception(
            telemetry,
            door_observation_origin(telemetry),
            np.zeros(2, dtype=np.float32),
            np.zeros(DOOR_PHASE_DIM, dtype=np.float32),
            np.zeros(DOOR_EVIDENCE_DIM, dtype=np.float32),
        )
