from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from flightrl.puffer4_door_policy import DOOR_OBS_DIM, FlightRLDoorEncoder
from flightrl.puffer4_door_policy_contract import DoorPolicyArchitecture


class _MinGRU(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, 3 * hidden_size, bias=False)]
        )

    @staticmethod
    def _g(values: torch.Tensor) -> torch.Tensor:
        return torch.where(values >= 0, values + 0.5, values.sigmoid())

    def initial_state(self, batch_size: int) -> tuple[torch.Tensor]:
        return (torch.zeros(1, batch_size, self.hidden_size),)

    def forward_eval(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        hidden, gate, projection = self.layers[0](inputs).chunk(3, dim=-1)
        recurrent = torch.lerp(state[0][0], self._g(hidden), gate.sigmoid())
        output = projection.sigmoid() * recurrent + (
            1.0 - projection.sigmoid()
        ) * inputs
        return output, (recurrent.unsqueeze(0),)


class _DoorDecoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.decoder_logstd = nn.Parameter(torch.zeros(1, 2))
        self.decoder_mean = nn.Linear(hidden_size, 2)
        self.value_function = nn.Linear(hidden_size, 1)


class DoorPufferRuntime(nn.Module):
    """Inference-only equivalent of the fixed-door Puffer policy."""

    observation_size = DOOR_OBS_DIM

    def __init__(self, hidden_size: int = 96) -> None:
        super().__init__()
        self.encoder = FlightRLDoorEncoder(DOOR_OBS_DIM, hidden_size)
        self.decoder = _DoorDecoder(hidden_size)
        self.network = _MinGRU(hidden_size)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        architecture: DoorPolicyArchitecture | None = None,
    ) -> DoorPufferRuntime:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        return cls.from_state_dict(state, architecture=architecture)

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, torch.Tensor],
        architecture: DoorPolicyArchitecture | None = None,
    ) -> DoorPufferRuntime:
        inferred_hidden_size = int(state["encoder.fusion.0.weight"].shape[0])
        if architecture is not None and (
            architecture.num_layers != 1
            or architecture.hidden_size != inferred_hidden_size
        ):
            raise ValueError(
                "fixed-door checkpoint architecture does not match bundle"
            )
        hidden_size = (
            inferred_hidden_size
            if architecture is None
            else architecture.hidden_size
        )
        policy = cls(hidden_size)
        policy.load_state_dict(state, strict=True)
        policy.eval()
        return policy

    def initial_state(self, batch_size: int = 1) -> tuple[torch.Tensor]:
        return self.network.initial_state(batch_size)

    def forward_eval(
        self,
        observation: torch.Tensor,
        state: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor]]:
        encoded = self.encoder(observation)
        hidden, next_state = self.network.forward_eval(encoded, state)
        return (
            self.decoder.decoder_mean(hidden),
            self.decoder.value_function(hidden),
            next_state,
        )
