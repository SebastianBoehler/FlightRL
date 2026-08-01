from __future__ import annotations

import torch

from flightrl.puffer4_door_asymmetric import (
    DoorAsymmetricCritic,
    actor_parameters,
    generalized_advantage,
    privileged_door_features,
)
from flightrl.puffer4_door_policy import (
    DOOR_OBS_DIM,
    DOOR_POLICY_OBS_DIM,
    FlightRLDoorEncoder,
)


def test_privileged_critic_features_exclude_pixels() -> None:
    first = torch.zeros(2, 3, DOOR_OBS_DIM)
    second = first.clone()
    second[..., :DOOR_POLICY_OBS_DIM] = 1.0
    second[..., DOOR_POLICY_OBS_DIM:] = first[..., DOOR_POLICY_OBS_DIM:]

    assert not torch.equal(
        privileged_door_features(first),
        privileged_door_features(second),
    )
    second[..., : 3 * 64 * 48] = 7.0
    second[..., 3 * 64 * 48 : DOOR_POLICY_OBS_DIM] = 0.0

    torch.testing.assert_close(
        privileged_door_features(first),
        privileged_door_features(second),
    )
    assert DoorAsymmetricCritic().input_dim == 32


def test_generalized_advantage_stops_at_terminal() -> None:
    rewards = torch.tensor(((1.0,), (2.0,), (100.0,)))
    values = torch.zeros_like(rewards)
    terminals = torch.tensor(((0.0,), (1.0,), (0.0,)))

    advantage, returns = generalized_advantage(
        rewards,
        values,
        terminals,
        torch.zeros(1),
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(advantage[:, 0], torch.tensor((3.0, 2.0, 100.0)))
    torch.testing.assert_close(returns, advantage)


def test_actor_optimizer_excludes_frozen_grounder_and_shared_value_head() -> None:
    class Decoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder_mean = torch.nn.Linear(32, 2)
            self.decoder_logstd = torch.nn.Parameter(torch.zeros(1, 2))
            self.value_function = torch.nn.Linear(32, 1)

    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = FlightRLDoorEncoder(DOOR_OBS_DIM, 32)
            self.network = torch.nn.Linear(32, 32)
            self.decoder = Decoder()

    policy = Policy()
    for parameter in policy.encoder.grounder.parameters():
        parameter.requires_grad_(False)
    selected = {id(parameter) for parameter in actor_parameters(policy)}

    assert id(policy.decoder.decoder_mean.weight) in selected
    assert id(policy.decoder.decoder_logstd) not in selected
    assert id(policy.decoder.value_function.weight) not in selected
    assert all(
        id(parameter) not in selected
        for parameter in policy.encoder.grounder.parameters()
    )
