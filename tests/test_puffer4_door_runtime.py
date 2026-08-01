from __future__ import annotations

import numpy as np
import pytest
import torch

from flightrl.puffer4_door_policy import DOOR_OBS_DIM
from flightrl.puffer4_door_runtime import (
    DoorFrameEncoder,
    DoorMissionPhase,
    DoorPufferRuntime,
    DoorPufferShadow,
)
from flightrl.semantic.contract import GroundingDetection, NormalizedBox


def _telemetry() -> dict[str, float]:
    return {
        "stateEstimate.x": 0.0,
        "stateEstimate.y": 0.0,
        "stateEstimate.z": 0.8,
        "stateEstimate.vx": 0.0,
        "stateEstimate.vy": 0.0,
        "stateEstimate.vz": 0.0,
        "stateEstimate.roll": 0.0,
        "stateEstimate.pitch": 0.0,
        "stateEstimate.yaw": 0.0,
        "gyro.x": 0.0,
        "gyro.y": 0.0,
        "gyro.z": 0.0,
    }


def _detection(scale: float) -> GroundingDetection:
    half = 0.5 * scale
    return GroundingDetection(
        "door",
        0.9,
        NormalizedBox(0.5 - half, 0.5 - half, 0.5 + half, 0.5 + half),
    )


def test_frame_encoder_matches_native_door_contract() -> None:
    encoder = DoorFrameEncoder()
    first = encoder.encode(np.full((96, 128), 51, dtype=np.uint8))
    second = encoder.encode(np.full((48, 64), 68, dtype=np.uint8))
    third = encoder.encode(np.full((48, 64), 51, dtype=np.uint8))
    pixels = 64 * 48

    assert first.shape == (3 * pixels,)
    np.testing.assert_allclose(first[:pixels], 0.2)
    np.testing.assert_allclose(first[pixels:], 0.0)
    np.testing.assert_allclose(second[:pixels], 68.0 / 255.0)
    np.testing.assert_allclose(second[pixels : 2 * pixels], 17.0 / 255.0)
    np.testing.assert_allclose(second[2 * pixels :], 0.0)
    np.testing.assert_allclose(third[pixels : 2 * pixels], -17.0 / 255.0)
    np.testing.assert_allclose(third[2 * pixels :], 0.0)


def test_frame_encoder_quantizes_before_mean_filling_airframe_mask() -> None:
    frame = np.full((48, 64), 25, dtype=np.uint8)
    frame[:12] = 8

    encoded = DoorFrameEncoder().encode(frame)

    assert encoded[0] == pytest.approx(12.0 / 255.0)
    assert encoded[24 * 64 + 32] == pytest.approx(17.0 / 255.0)


def test_mission_phase_tracks_search_track_approach_recover() -> None:
    phase = DoorMissionPhase(approach_scale=0.55)

    assert phase.update(None).name == "search"
    assert phase.update(_detection(0.4)).name == "track"
    assert phase.update(_detection(0.6)).name == "approach"
    assert phase.update(None).name == "recover"


def test_runtime_loads_checkpoint_and_shadow_outputs_finite_actions(tmp_path) -> None:
    checkpoint = tmp_path / "door.bin"
    source = DoorPufferRuntime(hidden_size=32)
    torch.save(source.state_dict(), checkpoint)
    shadow = DoorPufferShadow(checkpoint)

    result = shadow.step(
        np.full((48, 64), 51, dtype=np.uint8),
        _telemetry(),
        detection=None,
    )

    assert source.encoder.fusion[0].in_features == 174
    assert shadow.policy.encoder.fusion[0].in_features == 174
    assert result["phase"] == "search"
    assert result["monitor_only"] is True
    assert result["controls_drone"] is False
    assert 0.0 <= result["action_forward"] <= 1.0
    assert -1.0 <= result["action_yaw"] <= 1.0
    assert np.isfinite(result["inference_ms"])


def test_shadow_requires_explicit_reset_before_new_origin(tmp_path) -> None:
    checkpoint = tmp_path / "door.bin"
    torch.save(DoorPufferRuntime(hidden_size=32).state_dict(), checkpoint)
    shadow = DoorPufferShadow(checkpoint)
    shadow.step(np.zeros((48, 64), dtype=np.uint8), _telemetry(), detection=None)
    moved = _telemetry() | {"stateEstimate.x": 1.0}
    shadow.step(np.zeros((48, 64), dtype=np.uint8), moved, detection=None)

    shadow.reset()

    assert shadow.origin is None
    assert shadow.previous_action == pytest.approx((0.0, 0.0))
    assert shadow.policy.observation_size == DOOR_OBS_DIM
