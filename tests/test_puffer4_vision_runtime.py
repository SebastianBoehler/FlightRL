from __future__ import annotations

import numpy as np
import torch

from flightrl.puffer4_vision_runtime import (
    VisualObservationEncoder,
    VisualPufferRuntime,
)
from flightrl.puffer4_vision_policy import FlightRLVisionEncoder


def test_real_frame_encoder_matches_visual_contract() -> None:
    encoder = VisualObservationEncoder(3 * 16 * 12 + 6)
    frame = np.full((48, 64), 51, dtype=np.uint8)
    intent = np.asarray((1.0, 0.0, 0.0, 0.75, 0.0, 1.0), dtype=np.float32)

    first = encoder.encode(frame, intent)
    second = encoder.encode(frame, intent)

    assert first.shape == (3 * 16 * 12 + 6,)
    np.testing.assert_allclose(first[-6:], intent)
    np.testing.assert_allclose(first[: 3 * 16 * 12], 0.0)
    np.testing.assert_allclose(second, first)


def test_runtime_policy_recurrent_state_and_actions_are_finite() -> None:
    policy = VisualPufferRuntime(3 * 16 * 12 + 6, hidden_size=32)
    observation = torch.zeros((1, 3 * 16 * 12 + 6))

    action, value, state = policy.forward_eval(
        observation,
        policy.initial_state(),
    )

    assert action.shape == (1, 4)
    assert value.shape == (1, 1)
    assert state[0].shape == (1, 1, 32)
    assert torch.isfinite(action).all()


def test_training_only_privileged_label_is_invisible_to_encoder() -> None:
    encoder = FlightRLVisionEncoder(3 * 16 * 12 + 6 + 1, hidden_size=32)
    observations = torch.zeros((2, 3 * 16 * 12 + 6 + 1))
    observations[:, -1] = torch.tensor((-1.0, 1.0))

    encoded = encoder(observations)

    torch.testing.assert_close(encoded[0], encoded[1])
