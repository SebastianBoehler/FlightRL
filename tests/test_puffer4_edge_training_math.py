from __future__ import annotations

import numpy as np
import torch

import flightrl.puffer4_edge_training as edge_training
from flightrl.puffer4_edge_training_data import (
    weighted_smooth_l1_minimizer,
)
from puffer4_edge_training_support import training_dataset


def test_weighted_huber_constant_uses_minimizer_instead_of_mean() -> None:
    target = torch.tensor(((-1.0,), (1.0,), (1.0,)))
    weights = torch.ones(3)

    optimum = weighted_smooth_l1_minimizer(target, weights)

    assert optimum.item() == 0.5
    optimum_loss = torch.nn.functional.smooth_l1_loss(
        optimum.expand_as(target),
        target,
    )
    mean_loss = torch.nn.functional.smooth_l1_loss(
        target.mean(0).expand_as(target),
        target,
    )
    assert optimum_loss < mean_loss


def test_training_decision_weights_allow_all_critical_singletons() -> None:
    dataset = training_dataset("train", 11)
    dataset.resets[:] = 1
    dataset.dones[:-1] = 1
    dataset.episode_ids[:] = np.arange(
        dataset.episode_ids.size,
        dtype=np.uint64,
    ).reshape(dataset.shape)
    dataset.scene_group_ids[:] = np.where(
        dataset.grounding[..., 0] > 0.5,
        0,
        64,
    ).astype(np.uint8)

    weights = edge_training.edge_sequence_loss_weights(dataset)

    assert bool(weights.critical.all())
    assert torch.equal(weights.training_decision, weights.episode)
