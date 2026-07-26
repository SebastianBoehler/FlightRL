from __future__ import annotations

import torch

from flightrl.sixdof.puffer_ppo import PufferPpoConfig, puffer_transfer_replay_mse


class TailEncoder(torch.nn.Module):
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[:, -4:]


class IdentityNetwork(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class TailDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(4))

    def mean_action(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.bias


class PreviousActionPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TailEncoder()
        self.network = IdentityNetwork()
        self.decoder = TailDecoder()


def replay(sequence_start: bool) -> dict[str, torch.Tensor]:
    observations = torch.zeros(3, 28)
    observations[1:, -4:] = 1.0
    data = {
        "observations": observations,
        "target_actions": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ),
    }
    if sequence_start:
        data["sequence_start"] = torch.tensor([True, False, False])
    return data


def test_transfer_replay_autoregressive_loss_catches_previous_action_feedback() -> None:
    policy = PreviousActionPolicy()
    config = PufferPpoConfig(transfer_replay_coef=1.0, transfer_replay_batch_size=3)

    static_only = puffer_transfer_replay_mse(policy, replay(sequence_start=False), config)
    autoregressive = puffer_transfer_replay_mse(policy, replay(sequence_start=True), config)

    assert autoregressive > static_only
