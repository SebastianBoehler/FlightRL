from __future__ import annotations

import pytest
import torch

from flightrl.exploration.range_policy import RangeExplorationActorCritic


def test_range_actor_outputs_only_bounded_forward_and_yaw() -> None:
    torch.manual_seed(501)
    model = RangeExplorationActorCritic(hidden_size=64)
    observation = torch.zeros((3, 4106), dtype=torch.float32)
    action, value = model.forward_step(observation)

    assert action.shape == (3, 2)
    assert torch.all((0.0 <= action[:, 0]) & (action[:, 0] <= 1.0))
    assert torch.all((-1.0 <= action[:, 1]) & (action[:, 1] <= 1.0))
    assert value.shape == (3,)
    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000


def test_range_actor_yaw_has_real_gradient_path_from_exploration_map() -> None:
    torch.manual_seed(502)
    model = RangeExplorationActorCritic(hidden_size=64)
    observation = torch.linspace(0.0, 1.0, 4106).repeat(2, 1).float()
    observation.requires_grad_(True)

    action, _value = model.forward_step(observation)
    action[:, 1].sum().backward()

    assert observation.grad is not None
    assert float(observation.grad[:, :4096].abs().sum()) > 0.0


def test_range_actor_exposes_no_hidden_recurrent_state() -> None:
    model = RangeExplorationActorCritic(hidden_size=64)
    observation = torch.zeros((1, 4106), dtype=torch.float32)

    assert any(isinstance(module, torch.nn.GRUCell) for module in model.modules())
    with pytest.raises(TypeError):
        model.forward_step(observation, torch.zeros((1, 64)))


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (torch.zeros((2, 4105)), "observation shape"),
        (torch.zeros((2, 4106), dtype=torch.float64), "float32"),
        (torch.full((2, 4106), float("nan")), "finite"),
    ],
)
def test_range_actor_rejects_incompatible_runtime_tensors(
    observation: torch.Tensor,
    message: str,
) -> None:
    model = RangeExplorationActorCritic(hidden_size=64)
    with pytest.raises(ValueError, match=message):
        model.forward_step(observation)
