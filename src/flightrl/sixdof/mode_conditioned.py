from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn

from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy


MODES = ("obstacle_hover", "velocity_target")
BASE_OBSERVATION_DIM = 28


def mode_index(mode: str, modes: tuple[str, ...] = MODES) -> int:
    if mode not in modes:
        raise ValueError(f"unknown mode {mode!r}; expected one of {', '.join(modes)}")
    return modes.index(mode)


def append_mode_np(observations: np.ndarray, mode: str, modes: tuple[str, ...] = MODES) -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float32)
    one_hot = np.zeros((obs.shape[0], len(modes)), dtype=np.float32)
    one_hot[:, mode_index(mode, modes)] = 1.0
    return np.concatenate([obs, one_hot], axis=1).astype(np.float32)


def append_mode_torch(observations: torch.Tensor, mode: str, modes: tuple[str, ...] = MODES) -> torch.Tensor:
    obs = observations.float()
    one_hot = torch.zeros((obs.shape[0], len(modes)), dtype=obs.dtype, device=obs.device)
    one_hot[:, mode_index(mode, modes)] = 1.0
    return torch.cat([obs, one_hot], dim=1)


class ModeConditionedWrapper(nn.Module):
    def __init__(self, policy: PufferSixDofPolicy, mode: str, modes: tuple[str, ...] = MODES) -> None:
        super().__init__()
        self.policy = policy
        self.mode = mode
        self.modes = modes
        self.metadata = replace(policy.metadata, observation_dim=BASE_OBSERVATION_DIM)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.policy(append_mode_torch(observations, self.mode, self.modes))


def expand_policy_for_modes(base: PufferSixDofPolicy, modes: tuple[str, ...] = MODES) -> PufferSixDofPolicy:
    metadata = PufferPolicyMetadata(
        observation_dim=base.metadata.observation_dim + len(modes),
        hidden_size=base.metadata.hidden_size,
        action_dim=base.metadata.action_dim,
        num_layers=base.metadata.num_layers,
    )
    expanded = PufferSixDofPolicy(metadata)
    state = base.state_dict()
    expanded_state = expanded.state_dict()
    for key, value in state.items():
        if key == "encoder.encoder.weight":
            expanded_state[key][:, : value.shape[1]] = value
            expanded_state[key][:, value.shape[1] :] = 0.0
        else:
            expanded_state[key] = value
    expanded.load_state_dict(expanded_state)
    return expanded
