from __future__ import annotations

import numpy as np
import pytest

from flightrl.vision import (
    VisionObservationBatchEncoder,
    VisionObservationConfig,
    VisionObservationEncoder,
    append_vision_observation,
)


def test_default_contract_is_small_grayscale_image() -> None:
    config = VisionObservationConfig()

    assert config.shape == (1, 48, 64)
    assert config.flat_dim == 3072
    assert config.metadata()["normalization"] == "minus_one_one"


def test_temporal_channels_keep_appearance_and_add_change() -> None:
    config = VisionObservationConfig(
        width=2,
        height=2,
        frame_stack=2,
        include_delta=True,
        include_motion_mask=True,
        motion_threshold=0.25,
        normalization="zero_one",
    )
    encoder = VisionObservationEncoder(config)

    first = encoder.encode(np.zeros((2, 2), dtype=np.uint8))
    second = encoder.encode(np.full((2, 2), 255, dtype=np.uint8))

    assert first.shape == (4, 2, 2)
    assert np.all(first == 0.0)
    assert np.all(second[0] == 0.0)
    assert np.all(second[1] == 1.0)
    assert np.all(second[2] == 1.0)
    assert np.all(second[3] == 1.0)


def test_rgb_and_bgr_inputs_share_one_output_contract() -> None:
    rgb_config = VisionObservationConfig(width=1, height=1, normalization="zero_one")
    bgr_config = VisionObservationConfig(width=1, height=1, input_color_order="bgr", normalization="zero_one")

    rgb = VisionObservationEncoder(rgb_config).encode(np.array([[[255, 0, 0]]], dtype=np.uint8))
    bgr = VisionObservationEncoder(bgr_config).encode(np.array([[[0, 0, 255]]], dtype=np.uint8))

    assert rgb[0, 0, 0] == pytest.approx(0.299)
    assert np.allclose(rgb, bgr)


def test_batch_encoder_keeps_temporal_state_per_stream() -> None:
    config = VisionObservationConfig(width=1, height=1, include_delta=True, normalization="zero_one")
    encoder = VisionObservationBatchEncoder(config, batch_size=2)

    encoder.encode((np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)))
    batch = encoder.encode((np.full((1, 1), 255, dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)))

    assert batch.shape == (2, 2)
    assert np.allclose(batch[0], [1.0, 1.0])
    assert np.allclose(batch[1], [0.0, 0.0])


def test_append_vision_observation_supports_vectors_and_batches() -> None:
    vector = append_vision_observation(np.array([1.0, 2.0]), np.array([3.0]))
    batch = append_vision_observation(np.ones((2, 2)), np.zeros((2, 3)))

    assert np.allclose(vector, [1.0, 2.0, 3.0])
    assert batch.shape == (2, 5)


def test_contract_rejects_invalid_dimensions_and_pixels() -> None:
    with pytest.raises(ValueError, match="positive"):
        VisionObservationConfig(width=0)
    with pytest.raises(ValueError, match="motion_threshold"):
        VisionObservationConfig(motion_threshold=2.0)
    with pytest.raises(ValueError, match="pixels"):
        VisionObservationEncoder(VisionObservationConfig()).encode(np.array([[-1.0]], dtype=np.float32))
