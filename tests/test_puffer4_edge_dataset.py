from __future__ import annotations

import numpy as np
import pytest
import torch

from flightrl.puffer4_edge_dataset import (
    EDGE_STUDENT_OBSERVATION_DIM,
    EdgeTeacherRecord,
    adapt_native_door_observation,
    adapt_native_door_observation_batch,
    pack_gray4_nibbles,
    unpack_gray4_nibbles,
)
from flightrl.puffer4_edge_schema import (
    EDGE_FRAME_PIXELS,
    EDGE_OBSERVATION_DIM,
)


def _native_observation() -> np.ndarray:
    native = np.zeros(EDGE_STUDENT_OBSERVATION_DIM, dtype=np.float32)
    nibbles = np.arange(EDGE_FRAME_PIXELS, dtype=np.uint16) % 16
    native[:EDGE_FRAME_PIXELS] = nibbles / 15.0
    telemetry = native[EDGE_FRAME_PIXELS : EDGE_FRAME_PIXELS + 19]
    telemetry[:] = (
        0.1,
        -0.2,
        0.3,
        0.4,
        -0.5,
        0.6,
        0.0,
        0.0,
        1.0,
        0.4,
        0.2,
        -0.3,
        0.1,
        0.0,
        1.0,
        0.7,
        0.0,
        0.0,
        -0.6,
    )
    native[EDGE_FRAME_PIXELS + 19] = 1.0
    native[EDGE_OBSERVATION_DIM:] = (
        0.8,
        0.0,
        0.0,
        -0.25,
        1.0,
        -0.5,
        0.25,
        0.4,
    )
    return native


def test_gray4_codec_uses_even_high_odd_low_nibbles() -> None:
    nibbles = torch.arange(EDGE_FRAME_PIXELS, dtype=torch.int64) % 16
    normalized = nibbles.to(torch.float32) / 15.0

    packed = pack_gray4_nibbles(normalized)
    restored = unpack_gray4_nibbles(packed)

    assert isinstance(packed, bytes)
    assert len(packed) == EDGE_FRAME_PIXELS // 2
    assert packed[:3] == bytes((0x01, 0x23, 0x45))
    torch.testing.assert_close(restored, normalized, rtol=0.0, atol=0.0)


def test_native_adapter_preserves_exact_edge_prefix_and_training_tail() -> None:
    native = _native_observation()

    record = adapt_native_door_observation(
        native,
        target_id=0,
        reset=True,
        done_after_action=False,
    )
    model = record.model_observation()

    assert isinstance(record, EdgeTeacherRecord)
    assert record.reset is True
    assert record.done_after_action is False
    assert record.teacher_action == pytest.approx((0.8, 0.0, 0.0, -0.25))
    assert record.grounding == pytest.approx((1.0, -0.5, 0.25, 0.4))
    assert record.telemetry[15:] == pytest.approx((0.7, 0.0, 0.0, -0.6))
    assert model.shape == (EDGE_OBSERVATION_DIM,)
    assert model.dtype == np.float32
    np.testing.assert_array_equal(model, native[:EDGE_OBSERVATION_DIM])


def test_record_keeps_terminal_boundary_separate_from_next_reset() -> None:
    terminal = adapt_native_door_observation(
        _native_observation(),
        target_id=0,
        reset=False,
        done_after_action=True,
    )
    next_episode = adapt_native_door_observation(
        _native_observation(),
        target_id=0,
        reset=True,
        done_after_action=False,
    )

    assert terminal.done_after_action is True
    assert terminal.reset is False
    assert next_episode.done_after_action is False
    assert next_episode.reset is True


def test_adapter_rejects_non_gray4_or_mismatched_target() -> None:
    native = _native_observation()
    native[0] = 0.5
    with pytest.raises(ValueError, match="gray4"):
        adapt_native_door_observation(
            native,
            target_id=0,
            reset=False,
            done_after_action=False,
        )

    native = _native_observation()
    with pytest.raises(ValueError, match="target"):
        adapt_native_door_observation(
            native,
            target_id=1,
            reset=False,
            done_after_action=False,
        )


def test_adapter_rejects_invalid_absent_grounding_or_flags() -> None:
    native = _native_observation()
    native[-4:] = (0.0, 0.1, 0.0, 0.0)
    with pytest.raises(ValueError, match="absent"):
        adapt_native_door_observation(
            native,
            target_id=0,
            reset=False,
            done_after_action=False,
        )

    with pytest.raises(ValueError, match="boolean"):
        adapt_native_door_observation(
            _native_observation(),
            target_id=0,
            reset=1,
            done_after_action=False,
        )


def test_batch_adapter_derives_exact_native_target_ids() -> None:
    door = _native_observation()
    monitor = _native_observation()
    monitor[EDGE_FRAME_PIXELS + 19 : EDGE_OBSERVATION_DIM] = (0.0, 1.0, 0.0)

    batch = adapt_native_door_observation_batch(np.stack((door, monitor)))

    assert batch.target_ids.tolist() == [0, 1]
    assert batch.packed_frames.shape == (2, EDGE_FRAME_PIXELS // 2)
    np.testing.assert_array_equal(
        batch.teacher_actions[0], door[EDGE_OBSERVATION_DIM : EDGE_OBSERVATION_DIM + 4]
    )


@pytest.mark.parametrize("pixel", (-0.1, 1.1, 0.5))
def test_batch_adapter_rejects_out_of_range_or_non_gray4_pixels(pixel: float) -> None:
    native = _native_observation()
    native[0] = pixel

    with pytest.raises(ValueError, match="gray4"):
        adapt_native_door_observation_batch(native[None, :])


def test_batch_adapter_rejects_noncanonical_native_mission_token() -> None:
    native = _native_observation()
    native[EDGE_FRAME_PIXELS + 19 : EDGE_OBSERVATION_DIM] = (0.5, 0.5, 0.0)

    with pytest.raises(ValueError, match="one-hot"):
        adapt_native_door_observation_batch(native[None, :])
