from __future__ import annotations

import math

import pytest
import torch

from flightrl.bounded_action import BoundedNormal
from flightrl.sixdof.rl import SixDofActorCritic


def test_bounded_normal_samples_stay_inside_asymmetric_bounds() -> None:
    torch.manual_seed(3)
    distribution = BoundedNormal(
        torch.tensor([[30.0, -30.0], [-30.0, 30.0]]),
        torch.ones(2, 2),
        low=(0.0, -1.0),
        high=(1.0, 1.0),
    )

    actions, log_probability = distribution.rsample_with_log_prob()

    assert torch.all(actions[:, 0] >= 0.0)
    assert torch.all(actions[:, 0] <= 1.0)
    assert torch.all(actions[:, 1] >= -1.0)
    assert torch.all(actions[:, 1] <= 1.0)
    assert torch.isfinite(log_probability).all()


def test_bounded_normal_log_prob_has_tanh_and_affine_jacobian() -> None:
    distribution = BoundedNormal(
        torch.zeros(1, 1),
        torch.ones(1, 1),
        low=0.0,
        high=1.0,
    )

    actual = distribution.log_prob(torch.tensor([[0.5]]))
    base_at_zero = -0.5 * math.log(2.0 * math.pi)

    torch.testing.assert_close(
        actual,
        torch.tensor([base_at_zero - math.log(0.5)]),
    )


def test_bounded_normal_recomputes_identical_log_prob_at_boundaries() -> None:
    distribution = BoundedNormal(
        torch.zeros(2, 2),
        torch.full((2, 2), 0.4),
        low=(0.0, -1.0),
        high=(1.0, 1.0),
    )
    actions = torch.tensor([[0.0, -1.0], [1.0, 1.0]])

    first = distribution.log_prob(actions)
    second = distribution.log_prob(actions)

    assert torch.isfinite(first).all()
    torch.testing.assert_close(torch.exp(second - first), torch.ones(2))
    with pytest.raises(ValueError, match="outside"):
        distribution.log_prob(torch.tensor([[1.1, 0.0]]))


def test_bounded_normal_mode_transforms_location_into_action_space() -> None:
    distribution = BoundedNormal(
        torch.tensor([[0.0, math.atanh(0.5)]]),
        torch.ones(1, 2),
        low=(0.0, -1.0),
        high=(1.0, 1.0),
    )

    torch.testing.assert_close(distribution.mode, torch.tensor([[0.5, 0.5]]))


def test_bounded_normal_from_mode_preserves_deterministic_actor_output() -> None:
    expected = torch.tensor([[0.8, -0.4]])

    distribution = BoundedNormal.from_mode(
        expected,
        torch.ones(1, 2),
    )

    torch.testing.assert_close(distribution.mode, expected)


@pytest.mark.parametrize(
    ("location", "scale", "message"),
    (
        (torch.tensor([[float("nan")]]), torch.ones(1, 1), "location"),
        (torch.zeros(1, 1), torch.zeros(1, 1), "scale"),
        (torch.zeros(1, 1), torch.tensor([[float("inf")]]), "scale"),
    ),
)
def test_bounded_normal_rejects_nonfinite_or_nonpositive_parameters(
    location: torch.Tensor,
    scale: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BoundedNormal(location, scale)


def test_sixdof_actor_recomputes_saturated_rollout_log_probability() -> None:
    torch.manual_seed(5)
    actor = SixDofActorCritic(input_dim=28, hidden_size=16)
    with torch.no_grad():
        actor.actor.net[-2].weight.zero_()
        actor.actor.net[-2].bias.fill_(30.0)
        actor.log_std.fill_(math.log(2.0))
    observations = torch.randn(128, 28)

    actions, pre_tanh, old_log_probability, _entropy, _value = actor.act(
        observations,
        action_std=1.0,
    )
    new_log_probability, _entropy, _value = actor.evaluate_actions(
        observations,
        pre_tanh,
        action_std=1.0,
    )

    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)
    assert torch.min(pre_tanh) > 20.0
    assert torch.unique(pre_tanh[:, 0]).numel() > torch.unique(actions[:, 0]).numel()
    inverse_log_probability = BoundedNormal.from_mode(
        actor.actor(observations),
        torch.exp(actor.log_std).clamp(0.05, 2.0),
    ).log_prob(actions)
    assert not torch.allclose(inverse_log_probability, old_log_probability)
    torch.testing.assert_close(
        torch.exp(new_log_probability - old_log_probability),
        torch.ones(128),
    )
