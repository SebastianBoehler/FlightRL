from __future__ import annotations

import torch
from torch import nn

from flightrl.mujoco.semantic_observation import SemanticStudentObservationLayout
from flightrl.policy import MinGRU

RISK_CLEARANCE_M = 0.65
RISK_SHARPNESS_PER_M = 10.0


def _visual_encoder(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, 8, kernel_size=5, stride=4, padding=2),
        nn.GELU(),
        nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(12, 16, kernel_size=3, stride=2, padding=1),
        nn.GELU(),
        nn.AdaptiveAvgPool2d((3, 4)),
        nn.Flatten(),
    )


class VisualSafetyModel(nn.Module):
    feature_dim = 16 * 3 * 4

    def __init__(self, layout: SemanticStudentObservationLayout) -> None:
        super().__init__()
        self.layout = layout
        self.encoder = _visual_encoder(layout.vision.channels)
        self.clearance_head = nn.Linear(self.feature_dim, 1)
        self.collision_risk_head = nn.Linear(self.feature_dim, 1)

    def features(self, observations: torch.Tensor) -> torch.Tensor:
        values = observations.reshape(observations.shape[0], -1).float()
        images = values[:, self.layout.vision_slice].reshape(
            -1,
            *self.layout.vision.shape,
        )
        return self.encoder(images)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(observations)
        clearance_m = 4.0 * self.normalized_clearance(features)
        collision_risk = torch.sigmoid(self.collision_risk_head(features))
        return clearance_m, collision_risk

    def normalized_clearance(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.clearance_head(features))


class RecurrentVisualSafetyModel(nn.Module):
    feature_dim = VisualSafetyModel.feature_dim

    def __init__(
        self,
        layout: SemanticStudentObservationLayout,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.encoder = _visual_encoder(layout.vision.channels)
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.GELU(),
        )
        self.network = MinGRU(hidden_size)
        self.clearance_head = nn.Linear(hidden_size, 1)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor]:
        return self.network.initial_state(batch_size, device)

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor]]:
        hidden = self.projection(self._features(observations))
        hidden, next_state = self.network.forward_eval(hidden, state)
        clearance, risk = self._heads(hidden)
        return clearance, risk, next_state

    def forward_train(
        self,
        observations: torch.Tensor,
        *,
        batch: int,
        horizon: int,
        state: tuple[torch.Tensor] | None,
        terminals: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor] | None,
    ]:
        values = observations.reshape(observations.shape[0], -1).float()
        vision = values[:, self.layout.vision_slice].reshape(
            batch,
            horizon,
            *self.layout.vision.shape,
        )
        return self.forward_train_vision(
            vision,
            state=state,
            terminals=terminals,
        )

    def forward_train_vision(
        self,
        vision: torch.Tensor,
        *,
        state: tuple[torch.Tensor] | None,
        terminals: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor] | None,
    ]:
        batch, horizon = vision.shape[:2]
        features = self.encoder(
            vision.reshape(batch * horizon, *self.layout.vision.shape).float()
        )
        sequence = self.projection(features).reshape(batch, horizon, -1)
        if state is None:
            if terminals is None:
                recurrent = self.network.forward_train(sequence)
                next_state = None
            else:
                recurrent, next_state = self.network.forward_train_stateful_masked(
                    sequence,
                    self.initial_state(batch, vision.device),
                    terminals,
                )
        elif terminals is None:
            recurrent, next_state = self.network.forward_train_stateful(
                sequence,
                state,
            )
        else:
            recurrent, next_state = self.network.forward_train_stateful_masked(
                sequence,
                state,
                terminals,
            )
        clearance, risk = self._heads(recurrent.reshape(batch * horizon, -1))
        return clearance, risk, next_state

    def _features(self, observations: torch.Tensor) -> torch.Tensor:
        values = observations.reshape(observations.shape[0], -1).float()
        images = values[:, self.layout.vision_slice].reshape(
            -1,
            *self.layout.vision.shape,
        )
        return self.encoder(images)

    def _heads(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clearance = 4.0 * torch.sigmoid(self.clearance_head(hidden))
        return clearance, collision_risk_from_clearance(clearance)


class RecurrentSafetyModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.clearance_head = nn.Linear(hidden_size, 1)
        self.collision_risk_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        recurrent_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            4.0 * torch.sigmoid(self.clearance_head(recurrent_features)),
            torch.sigmoid(self.collision_risk_head(recurrent_features)),
        )


def collision_risk_from_clearance(clearance_m: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(collision_risk_logits_from_clearance(clearance_m))


def collision_risk_logits_from_clearance(
    clearance_m: torch.Tensor,
) -> torch.Tensor:
    return RISK_SHARPNESS_PER_M * (RISK_CLEARANCE_M - clearance_m)
