from __future__ import annotations

import torch


def replay_sample_weights(replay: dict[str, torch.Tensor], indices: torch.Tensor) -> torch.Tensor | None:
    weights = replay.get("sample_weights")
    if weights is None:
        return None
    return weights[indices].to(device=indices.device, dtype=torch.float32)


def weighted_mse_loss(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    per_sample = torch.mean((prediction - target).pow(2), dim=1)
    return weighted_mean(per_sample, weights)


def weighted_envelope_loss(prediction: torch.Tensor, action_abs_limit: float, weights: torch.Tensor | None) -> torch.Tensor:
    per_sample = torch.mean(torch.relu(torch.abs(prediction) - action_abs_limit).pow(2), dim=1)
    return weighted_mean(per_sample, weights)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return torch.mean(values)
    scaled = weights / torch.clamp(torch.mean(weights), min=1e-6)
    return torch.mean(values * scaled)
