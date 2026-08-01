from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class DoorRollout:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probabilities: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    initial_state: tuple[torch.Tensor, ...]


def generalized_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminals: torch.Tensor,
    bootstrap_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantage = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(bootstrap_value)
    next_value = bootstrap_value
    for step in range(rewards.shape[0] - 1, -1, -1):
        alive = 1.0 - terminals[step]
        delta = rewards[step] + gamma * next_value * alive - values[step]
        next_advantage = (
            delta + gamma * gae_lambda * alive * next_advantage
        )
        advantage[step] = next_advantage
        next_value = values[step]
    return advantage, advantage + values
