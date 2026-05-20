from __future__ import annotations

import numpy as np
import torch

from flightrl.sixdof.offline import compute_action_weights, dataset_loss
from flightrl.sixdof.policies import SixDofPolicy


def test_inverse_std_action_weights_prioritize_low_variance_channels() -> None:
    actions = np.asarray(
        [
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.1, 0.2, 1.0],
            [0.0, -0.1, -0.2, 0.0],
        ],
        dtype=np.float32,
    )

    weights = compute_action_weights(actions, "inverse_std")

    assert weights.shape == (4,)
    assert weights[0] > weights[3]
    assert abs(float(np.mean(weights)) - 1.0) < 1e-6


def test_weighted_dataset_loss_accepts_action_weights() -> None:
    model = SixDofPolicy(hidden_size=8, input_dim=28)
    observations = np.zeros((4, 28), dtype=np.float32)
    actions = np.zeros((4, 4), dtype=np.float32)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    loss = dataset_loss(model, observations, actions, batch_size=2, action_weights=np.ones(4, dtype=np.float32))

    assert loss == 0.0
