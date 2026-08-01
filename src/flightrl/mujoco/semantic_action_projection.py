from __future__ import annotations

import math

import torch

MEMORY_CENTERING_LIMIT_RAD = math.radians(20.0)
SEARCH_YAW_NORMALIZED = 20.0 / 60.0
TRACKING_YAW_NORMALIZED = 8.0 / 60.0


def project_semantic_actions(
    distribution: torch.distributions.Normal,
    *,
    action_mode: str,
    acquired: torch.Tensor,
    memory_bearing: torch.Tensor,
    confidence: torch.Tensor,
    horizontal_error: torch.Tensor,
    clearance_m: torch.Tensor | None,
    collision_risk: torch.Tensor | None,
) -> torch.distributions.Normal:
    search_yaw = distribution.mean[:, 3:].clamp(
        -SEARCH_YAW_NORMALIZED,
        SEARCH_YAW_NORMALIZED,
    )
    memory_yaw = (memory_bearing / MEMORY_CENTERING_LIMIT_RAD).clamp(
        -1.0, 1.0
    ) * SEARCH_YAW_NORMALIZED
    tracking_yaw = (-0.5 * horizontal_error).clamp(
        -TRACKING_YAW_NORMALIZED,
        TRACKING_YAW_NORMALIZED,
    )
    yaw = torch.where(acquired > 0.0, memory_yaw, search_yaw)
    yaw = torch.where(confidence > 0.0, tracking_yaw, yaw)
    if action_mode == "active_exploration":
        return _project_active_actions(
            distribution,
            acquired=acquired,
            memory_bearing=memory_bearing,
            confidence=confidence,
            horizontal_error=horizontal_error,
            clearance_m=clearance_m,
            collision_risk=collision_risk,
            yaw=yaw,
        )
    mean = torch.cat((distribution.mean[:, :3] * acquired, yaw), dim=1)
    return torch.distributions.Normal(mean, distribution.stddev)


def _project_active_actions(
    distribution: torch.distributions.Normal,
    *,
    acquired: torch.Tensor,
    memory_bearing: torch.Tensor,
    confidence: torch.Tensor,
    horizontal_error: torch.Tensor,
    clearance_m: torch.Tensor | None,
    collision_risk: torch.Tensor | None,
    yaw: torch.Tensor,
) -> torch.distributions.Normal:
    if clearance_m is None or collision_risk is None:
        raise RuntimeError("active action projection requires visual safety estimates")
    clearance_gate = torch.sigmoid(10.0 * (clearance_m - 0.65))
    risk_gate = torch.sigmoid(16.0 * (0.35 - collision_risk))
    safety_gate = clearance_gate * risk_gate
    learned_forward = torch.sigmoid(distribution.mean[:, :1]) * safety_gate
    target_visible = confidence > 0.0
    target_centered = target_visible & (torch.abs(horizontal_error) <= 0.5)
    forward = learned_forward
    forward = torch.where(
        target_visible & ~target_centered,
        torch.zeros_like(forward),
        forward,
    )
    memory_centered = (acquired > 0.0) & (
        torch.abs(memory_bearing) <= MEMORY_CENTERING_LIMIT_RAD
    )
    memory_only = (acquired > 0.0) & ~target_visible
    forward = torch.where(
        memory_only & ~memory_centered,
        torch.zeros_like(forward),
        forward,
    )
    mean = torch.cat(
        (
            forward,
            torch.zeros_like(distribution.mean[:, 1:3]),
            yaw,
        ),
        dim=1,
    )
    stddev = torch.cat(
        (
            distribution.stddev[:, :1],
            torch.full_like(distribution.stddev[:, 1:3], 1e-4),
            distribution.stddev[:, 3:],
        ),
        dim=1,
    )
    return torch.distributions.Normal(mean, stddev)
