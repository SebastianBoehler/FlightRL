from __future__ import annotations

import torch


def weighted_smooth_l1_minimizer(
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if target.ndim < 2 or weights.shape != target.shape[:-1]:
        raise ValueError("weighted Huber target and weights have incompatible shapes")
    if not torch.isfinite(target).all() or not torch.isfinite(weights).all():
        raise ValueError("weighted Huber target and weights must be finite")
    if torch.any(weights < 0.0) or not bool(torch.any(weights > 0.0)):
        raise ValueError("weighted Huber weights must be nonnegative with positive mass")
    values = target.reshape(-1, target.shape[-1]).to(torch.float64)
    mass = weights.reshape(-1, 1).to(torch.float64)
    active = mass[:, 0] > 0.0
    lower = values[active].amin(dim=0)
    upper = values[active].amax(dim=0)
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        derivative = (mass * (midpoint - values).clamp(-1.0, 1.0)).sum(0)
        lower = torch.where(derivative <= 0.0, midpoint, lower)
        upper = torch.where(derivative >= 0.0, midpoint, upper)
    return (0.5 * (lower + upper)).to(dtype=target.dtype)
