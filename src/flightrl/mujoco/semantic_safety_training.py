from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from flightrl.mujoco.semantic_safety_encoder import RecurrentVisualSafetyModel
from flightrl.mujoco.semantic_safety_replay import (
    BalancedSafetyReplay,
    SafetyReplayConfig,
    safety_supervision_losses,
)


@dataclass(frozen=True, slots=True)
class SafetyStepMetrics:
    clearance_loss: float
    collision_risk_loss: float
    replay_clearance_loss: float | None
    replay_collision_risk_loss: float | None


class RecurrentSafetyBootstrap:
    def __init__(
        self,
        model: RecurrentVisualSafetyModel,
        *,
        learning_rate: float,
        clearance_loss_scale: float,
        collision_risk_loss_scale: float,
        seed: int,
    ) -> None:
        self.model = model
        self.clearance_loss_scale = clearance_loss_scale
        self.collision_risk_loss_scale = collision_risk_loss_scale
        self.optimizer = torch.optim.AdamW(
            tuple(model.parameters()),
            lr=learning_rate,
        )
        self.replay = BalancedSafetyReplay(
            vision_slice=model.layout.vision_slice,
            vision_shape=model.layout.vision.shape,
            seed=seed,
        )
        self.replay_clearance_losses: list[float] = []
        self.replay_collision_risk_losses: list[float] = []

    @property
    def parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self.model.parameters())

    @property
    def config(self) -> SafetyReplayConfig:
        return self.replay.config

    def step(
        self,
        *,
        update: int,
        predicted_clearance_m: torch.Tensor,
        target_clearance_m: torch.Tensor,
        observations: np.ndarray,
        state_resets: np.ndarray,
    ) -> SafetyStepMetrics:
        online_clearance, online_risk = safety_supervision_losses(
            predicted_clearance_m,
            target_clearance_m,
        )
        self.replay.add(
            observations,
            target_clearance_m.detach().cpu().numpy(),
            state_resets,
        )
        replay_batch = self.replay.sample(update=update)
        total = self._scaled(online_clearance, online_risk)
        replay_clearance = None
        replay_risk = None
        if replay_batch is not None:
            replay_prediction, _, _ = self.model.forward_train_vision(
                replay_batch.vision,
                state=None,
                terminals=replay_batch.state_resets,
            )
            replay_clearance, replay_risk = safety_supervision_losses(
                replay_prediction,
                replay_batch.clearance_m,
                replay_batch.loss_mask,
            )
            total = total + self._scaled(replay_clearance, replay_risk)
        self.optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
        self.optimizer.step()
        if replay_clearance is not None and replay_risk is not None:
            self.replay_clearance_losses.append(float(replay_clearance.detach()))
            self.replay_collision_risk_losses.append(float(replay_risk.detach()))
        return SafetyStepMetrics(
            clearance_loss=float(online_clearance.detach()),
            collision_risk_loss=float(online_risk.detach()),
            replay_clearance_loss=_optional_float(replay_clearance),
            replay_collision_risk_loss=_optional_float(replay_risk),
        )

    def _scaled(
        self,
        clearance_loss: torch.Tensor,
        collision_risk_loss: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.clearance_loss_scale * clearance_loss
            + self.collision_risk_loss_scale * collision_risk_loss
        )


def _optional_float(value: torch.Tensor | None) -> float | None:
    return None if value is None else float(value.detach())
