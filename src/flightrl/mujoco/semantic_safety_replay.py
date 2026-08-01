from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional

from flightrl.mujoco.semantic_safety_encoder import (
    RISK_CLEARANCE_M,
    collision_risk_logits_from_clearance,
)

SIDE_CLEARANCE_OFFSET_M = 0.20


@dataclass(frozen=True, slots=True)
class SafetyReplayConfig:
    capacity_per_class: int = 12
    samples_per_class: int = 2
    additions_per_class: int = 2
    burn_in_steps: int = 8
    replay_interval: int = 4
    danger_clearance_m: float = RISK_CLEARANCE_M
    safe_clearance_m: float = 0.90

    def __post_init__(self) -> None:
        counts = (
            self.capacity_per_class,
            self.samples_per_class,
            self.additions_per_class,
            self.replay_interval,
        )
        if min(counts) <= 0 or self.burn_in_steps < 0:
            raise ValueError("replay sizes and interval must be positive")
        if self.safe_clearance_m <= self.danger_clearance_m:
            raise ValueError("safe clearance must exceed danger clearance")


@dataclass(frozen=True, slots=True)
class SafetyReplayBatch:
    vision: torch.Tensor
    clearance_m: torch.Tensor
    state_resets: torch.Tensor
    loss_mask: torch.Tensor
    is_danger: torch.Tensor


@dataclass(frozen=True, slots=True)
class _StoredSequence:
    vision: torch.Tensor
    clearance_m: torch.Tensor
    state_resets: torch.Tensor


class BalancedSafetyReplay:
    def __init__(
        self,
        *,
        vision_slice: slice,
        vision_shape: tuple[int, int, int],
        config: SafetyReplayConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.vision_slice = vision_slice
        self.vision_shape = vision_shape
        self.config = config or SafetyReplayConfig()
        self.rng = np.random.default_rng(seed)
        capacity = self.config.capacity_per_class
        self._danger: deque[_StoredSequence] = deque(maxlen=capacity)
        self._safe: deque[_StoredSequence] = deque(maxlen=capacity)

    @property
    def counts(self) -> dict[str, int]:
        return {"danger": len(self._danger), "safe": len(self._safe)}

    def add(
        self,
        observations: np.ndarray,
        clearance_m: np.ndarray,
        state_resets: np.ndarray,
    ) -> None:
        values = np.asarray(observations, dtype=np.float32)
        clearances = np.asarray(clearance_m, dtype=np.float32)
        resets = np.asarray(state_resets, dtype=np.float32)
        if values.ndim != 3 or clearances.shape != values.shape[:2]:
            raise ValueError("replay expects [batch, time, observation] sequences")
        if resets.shape != clearances.shape:
            raise ValueError("state reset shape must match clearance shape")
        vision = values[..., self.vision_slice].reshape(
            values.shape[0],
            values.shape[1],
            *self.vision_shape,
        )
        minimum = np.min(clearances, axis=1)
        self._add_selected(
            self._danger,
            np.flatnonzero(minimum < self.config.danger_clearance_m),
            vision,
            clearances,
            resets,
        )
        self._add_selected(
            self._safe,
            np.flatnonzero(minimum >= self.config.safe_clearance_m),
            vision,
            clearances,
            resets,
        )

    def sample(self, *, update: int) -> SafetyReplayBatch | None:
        if update % self.config.replay_interval != 0:
            return None
        count = min(
            self.config.samples_per_class,
            len(self._danger),
            len(self._safe),
        )
        if count == 0:
            return None
        danger = self._sample_bucket(self._danger, count)
        safe = self._sample_bucket(self._safe, count)
        stored = danger + safe
        horizon = stored[0].clearance_m.shape[0]
        burn_in = min(self.config.burn_in_steps, max(0, horizon - 1))
        loss_mask = torch.ones((2 * count, horizon), dtype=torch.float32)
        loss_mask[:, :burn_in] = 0.0
        return SafetyReplayBatch(
            vision=torch.stack([item.vision for item in stored]).float(),
            clearance_m=torch.stack(
                [item.clearance_m for item in stored]
            ).float(),
            state_resets=torch.stack(
                [item.state_resets for item in stored]
            ).float(),
            loss_mask=loss_mask,
            is_danger=torch.tensor([True] * count + [False] * count),
        )

    def _add_selected(
        self,
        bucket: deque[_StoredSequence],
        indices: np.ndarray,
        vision: np.ndarray,
        clearances: np.ndarray,
        resets: np.ndarray,
    ) -> None:
        limit = self.config.additions_per_class
        if len(indices) > limit:
            indices = self.rng.choice(indices, size=limit, replace=False)
        for index in indices:
            bucket.append(
                _StoredSequence(
                    vision=torch.from_numpy(
                        np.ascontiguousarray(vision[index])
                    ).half(),
                    clearance_m=torch.from_numpy(
                        np.ascontiguousarray(clearances[index])
                    ).half(),
                    state_resets=torch.from_numpy(
                        np.ascontiguousarray(resets[index])
                    ).bool(),
                )
            )

    def _sample_bucket(
        self,
        bucket: deque[_StoredSequence],
        count: int,
    ) -> list[_StoredSequence]:
        indices = self.rng.choice(len(bucket), size=count, replace=False)
        return [bucket[int(index)] for index in indices]


def action_corridor_clearance(ranges_m: np.ndarray) -> np.ndarray:
    ranges = np.asarray(ranges_m, dtype=np.float32)
    if ranges.ndim != 2 or ranges.shape[1] < 4:
        raise ValueError("ranges must contain front, back, left, and right")
    side = np.minimum(ranges[:, 2], ranges[:, 3])
    return np.minimum(ranges[:, 0], side + SIDE_CLEARANCE_OFFSET_M)


def safety_supervision_losses(
    predicted_clearance_m: torch.Tensor,
    target_clearance_m: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted = predicted_clearance_m.reshape(target_clearance_m.shape)
    target = target_clearance_m.float()
    mask = torch.ones_like(target) if loss_mask is None else loss_mask.float()
    clearance_weights = 1.0 + 4.0 * (target < 1.0).float()
    clearance_error = functional.smooth_l1_loss(
        predicted / 4.0,
        (target / 4.0).clamp(0.0, 1.0),
        reduction="none",
    )
    clearance_loss = _masked_mean(
        clearance_error * clearance_weights,
        mask,
    )

    danger = (target < RISK_CLEARANCE_M).float()
    valid_count = mask.sum().clamp_min(1.0)
    danger_count = (danger * mask).sum().clamp_min(1.0)
    safe_count = ((1.0 - danger) * mask).sum().clamp_min(1.0)
    danger_weight = (0.5 * valid_count / danger_count).clamp_max(20.0)
    safe_weight = (0.5 * valid_count / safe_count).clamp_max(20.0)
    class_weights = torch.where(danger > 0.0, danger_weight, safe_weight)
    risk_error = functional.binary_cross_entropy_with_logits(
        collision_risk_logits_from_clearance(predicted),
        danger,
        reduction="none",
    )
    risk_loss = _masked_mean(risk_error * class_weights, mask)
    return clearance_loss, risk_loss


def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * (weights > 0.0)).sum() / weights.sum().clamp_min(1.0)
