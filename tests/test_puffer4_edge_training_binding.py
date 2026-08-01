from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from flightrl.puffer4_edge_contract import EDGE_FRAME_PIXELS, EDGE_OBSERVATION_DIM
from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_training_selection import (
    cyclic_selection_frame_ablation,
    visual_dependence_check,
)
from flightrl.puffer4_edge_training_state import edge_state_dict_sha256


def test_visual_ablation_cyclically_permutes_only_selection_frames() -> None:
    observation = torch.arange(
        3 * EDGE_OBSERVATION_DIM,
        dtype=torch.float32,
    ).reshape(3, EDGE_OBSERVATION_DIM)

    ablated = cyclic_selection_frame_ablation(observation)

    assert torch.equal(
        ablated[:, :EDGE_FRAME_PIXELS],
        observation[:, :EDGE_FRAME_PIXELS].roll(1, dims=0),
    )
    assert torch.equal(
        ablated[:, EDGE_FRAME_PIXELS:],
        observation[:, EDGE_FRAME_PIXELS:],
    )
    assert torch.equal(
        ablated[:, :EDGE_FRAME_PIXELS].sort(dim=0).values,
        observation[:, :EDGE_FRAME_PIXELS].sort(dim=0).values,
    )
    assert torch.equal(observation[0], torch.arange(EDGE_OBSERVATION_DIM))


def test_visual_ablation_requires_two_independent_selection_agents() -> None:
    observation = torch.zeros(1, EDGE_OBSERVATION_DIM, dtype=torch.float32)

    with pytest.raises(ValueError, match="at least two selection agents"):
        cyclic_selection_frame_ablation(observation)


@pytest.mark.parametrize(
    ("clean", "ablated", "expected"),
    [
        (0.001, 0.0011, True),
        (0.001, 0.0010999, False),
        (1.0, 1.05, True),
        (1.0, 1.049999, False),
    ],
)
def test_visual_dependence_requires_absolute_or_relative_loss_increase(
    clean: float,
    ablated: float,
    expected: bool,
) -> None:
    assert visual_dependence_check(clean, ablated) is expected


def test_actor_state_digest_is_order_independent_and_tensor_exact() -> None:
    state = EdgeNavigationActor(hidden_size=48).state_dict()
    reverse = OrderedDict(reversed(tuple(state.items())))

    digest = edge_state_dict_sha256(state)

    assert edge_state_dict_sha256(reverse) == digest
    changed = OrderedDict((name, value.clone()) for name, value in state.items())
    first = sorted(changed)[0]
    changed[first].reshape(-1)[0] += 1.0
    assert edge_state_dict_sha256(changed) != digest


@pytest.mark.parametrize(
    "state",
    [
        {"x": torch.tensor([1.0], dtype=torch.float64)},
        {"x": torch.tensor([[1.0]], dtype=torch.float32)},
        {"y": torch.tensor([1.0], dtype=torch.float32)},
    ],
)
def test_actor_state_digest_binds_names_dtypes_and_shapes(state: dict) -> None:
    reference = {"x": torch.tensor([1.0], dtype=torch.float32)}

    assert edge_state_dict_sha256(state) != edge_state_dict_sha256(reference)
