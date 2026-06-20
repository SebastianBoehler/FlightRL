from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn


class PufferEncoder(nn.Module):
    def __init__(self, observation_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(observation_dim, hidden_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations.view(observations.shape[0], -1).float())


class PufferMlp(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
        self.net = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class PufferDecoder(nn.Module):
    def __init__(self, hidden_size: int, action_dim: int) -> None:
        super().__init__()
        self.decoder_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.decoder_mean = nn.Linear(hidden_size, action_dim)
        self.value_function = nn.Linear(hidden_size, 1)

    def mean_action(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder_mean(hidden)


@dataclass(frozen=True, slots=True)
class PufferPolicyMetadata:
    observation_dim: int
    hidden_size: int
    action_dim: int
    num_layers: int


class PufferSixDofPolicy(nn.Module):
    def __init__(self, metadata: PufferPolicyMetadata) -> None:
        super().__init__()
        self.metadata = metadata
        self.encoder = PufferEncoder(metadata.observation_dim, metadata.hidden_size)
        self.network = PufferMlp(metadata.hidden_size, metadata.num_layers)
        self.decoder = PufferDecoder(metadata.hidden_size, metadata.action_dim)

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, torch.Tensor]) -> "PufferSixDofPolicy":
        metadata = infer_metadata(state_dict)
        policy = cls(metadata)
        policy.load_state_dict(dict(state_dict))
        policy.eval()
        return policy

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.network(self.encoder(observations))
        return self.decoder.mean_action(hidden).clamp(-1.0, 1.0)

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.network(self.encoder(observations))
        return self.decoder.value_function(hidden).squeeze(-1)


def infer_metadata(state_dict: Mapping[str, torch.Tensor]) -> PufferPolicyMetadata:
    encoder = state_dict["encoder.encoder.weight"]
    decoder = state_dict["decoder.decoder_mean.weight"]
    layer_indices = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("network.net.") and key.endswith(".weight")
    }
    if not layer_indices:
        raise ValueError("Puffer checkpoint does not contain an MLP network")
    return PufferPolicyMetadata(
        observation_dim=int(encoder.shape[1]),
        hidden_size=int(encoder.shape[0]),
        action_dim=int(decoder.shape[0]),
        num_layers=len(layer_indices),
    )


def load_puffer_sixdof_policy(path: str) -> PufferSixDofPolicy:
    state_dict = torch.load(path, map_location="cpu")
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return PufferSixDofPolicy.from_state_dict(state_dict)
