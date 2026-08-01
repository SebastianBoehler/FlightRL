from __future__ import annotations

import torch
from torch import nn

from flightrl.puffer4_edge_action_contract import (
    EDGE_CONTROLLED_ACTION_AXES,
    EDGE_CONTROLLED_TELEMETRY_INDICES,
)
from flightrl.puffer4_edge_contract import (
    EDGE_ACTION_DIM,
    EDGE_FRAME_PIXELS,
    EDGE_HEIGHT,
    EDGE_MISSION_TOKEN_COUNT,
    EDGE_OBSERVATION_DIM,
    EDGE_TELEMETRY_BOUNDS,
    EDGE_TELEMETRY_DIM,
    EDGE_WIDTH,
)


class HardGatedRecurrentCell(nn.Module):
    """Small bounded recurrent cell composed of integer-friendly operations."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, 2 * hidden_size)
        self.recurrent_projection = nn.Linear(hidden_size, hidden_size)
        self.candidate_activation = nn.ReLU6()
        self.gate_activation = nn.Hardsigmoid()

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        candidate, gate = self.input_projection(inputs).chunk(2, dim=-1)
        candidate = self.candidate_activation(
            candidate + self.recurrent_projection(state)
        )
        gate = self.gate_activation(gate)
        return gate * state + (1.0 - gate) * candidate


class EdgeNavigationActor(nn.Module):
    """Edge-shaped PyTorch reference; C/int8 lowering is not implemented."""

    observation_size = EDGE_OBSERVATION_DIM
    action_size = EDGE_ACTION_DIM

    def __init__(self, hidden_size: int = 48) -> None:
        super().__init__()
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
        ):
            raise ValueError("AI Deck recurrent hidden size must be an integer")
        if not 32 <= hidden_size <= 64:
            raise ValueError("AI Deck recurrent hidden size must be in [32, 64]")
        self.hidden_size = hidden_size
        self.visual = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=4, padding=2),
            nn.ReLU6(),
            nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU6(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        visual_size = 12 * 3 * 4
        self.grounding_target_gate = nn.Sequential(
            nn.Linear(EDGE_MISSION_TOKEN_COUNT, visual_size),
            nn.Hardsigmoid(),
        )
        self.grounding_head = nn.Linear(visual_size, 4)
        fusion_size = (
            visual_size + EDGE_TELEMETRY_DIM + EDGE_MISSION_TOKEN_COUNT + 4
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, hidden_size),
            nn.ReLU6(),
        )
        self.recurrent = HardGatedRecurrentCell(hidden_size)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, len(EDGE_CONTROLLED_ACTION_AXES)),
            nn.Hardtanh(-1.0, 1.0),
        )
        nn.init.zeros_(self.action_head[0].weight)
        nn.init.zeros_(self.action_head[0].bias)
        self.visible_activation = nn.Hardsigmoid()
        self.center_activation = nn.Hardtanh(-1.0, 1.0)
        self.scale_activation = nn.Hardsigmoid()

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("edge batch size must be a positive integer")
        return torch.zeros(
            batch_size,
            self.hidden_size,
            dtype=torch.float32,
            device=device,
        )

    def forward_step(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action, grounding, _visibility_logit, next_state = (
            self.forward_training_step(observation, state)
        )
        return action, grounding, next_state

    def forward_training_step(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[1] != EDGE_OBSERVATION_DIM:
            raise ValueError(
                f"edge observation must have shape [batch, {EDGE_OBSERVATION_DIM}]"
            )
        if state.shape != (observation.shape[0], self.hidden_size):
            raise ValueError("edge recurrent state does not match batch or contract")
        if observation.dtype != torch.float32 or state.dtype != torch.float32:
            raise ValueError("edge observation and recurrent state must be float32")
        if observation.device != state.device:
            raise ValueError("edge observation and recurrent state must share a device")
        if not torch.isfinite(observation).all() or not torch.isfinite(state).all():
            raise ValueError("edge observation and recurrent state must be finite")
        if torch.any((state < 0.0) | (state > 6.0)):
            raise ValueError("edge recurrent state violates the [0, 6] invariant")
        frame_end = EDGE_FRAME_PIXELS
        telemetry_end = frame_end + EDGE_TELEMETRY_DIM
        frame = observation[:, :frame_end].reshape(
            -1,
            1,
            EDGE_HEIGHT,
            EDGE_WIDTH,
        )
        telemetry = observation[:, frame_end:telemetry_end]
        mission = observation[:, telemetry_end:]
        self._validate_contract_values(frame, telemetry, mission)
        visual = self.visual(frame)
        grounding, visibility_logit = self._grounding_with_logit(visual, mission)
        encoded = self.fusion(
            torch.cat((visual, telemetry, mission, grounding), dim=1)
        )
        next_state = self.recurrent(encoded, state)
        action_delta = self.action_head(next_state)
        previous_controlled = telemetry[
            :, EDGE_CONTROLLED_TELEMETRY_INDICES
        ]
        controlled = torch.clamp(
            previous_controlled + action_delta,
            min=-1.0,
            max=1.0,
        )
        structural_zero = torch.zeros_like(controlled[:, 0])
        action = torch.stack(
            (
                controlled[:, 0],
                structural_zero,
                structural_zero,
                controlled[:, 1],
            ),
            dim=1,
        )
        if not all(
            torch.isfinite(value).all()
            for value in (action, grounding, visibility_logit, next_state)
        ):
            raise RuntimeError("edge actor produced nonfinite outputs or state")
        return action, grounding, visibility_logit, next_state

    @staticmethod
    def _validate_contract_values(
        frame: torch.Tensor,
        telemetry: torch.Tensor,
        mission: torch.Tensor,
    ) -> None:
        if torch.any((frame < 0.0) | (frame > 1.0)):
            raise ValueError("edge frame values must be in [0, 1]")
        gray4 = frame * 15.0
        if not torch.allclose(gray4, gray4.round(), atol=1.0e-6, rtol=0.0):
            raise ValueError("edge frame values must be exact unpacked gray4 levels")
        for index, (low, high) in enumerate(EDGE_TELEMETRY_BOUNDS):
            if torch.any((telemetry[:, index] < low) | (telemetry[:, index] > high)):
                raise ValueError("edge telemetry violates normalized bounds")
        if torch.any((mission != 0.0) & (mission != 1.0)) or torch.any(
            mission.sum(dim=1) != 1.0
        ):
            raise ValueError("edge mission token must be canonical one-hot")
        body_up_norm = torch.linalg.vector_norm(telemetry[:, 6:9], dim=1)
        yaw_norm = torch.linalg.vector_norm(telemetry[:, 13:15], dim=1)
        if not torch.allclose(
            body_up_norm,
            torch.ones_like(body_up_norm),
            atol=1.0e-4,
            rtol=0.0,
        ):
            raise ValueError("edge body-up vector must have unit norm")
        if not torch.allclose(
            yaw_norm,
            torch.ones_like(yaw_norm),
            atol=1.0e-4,
            rtol=0.0,
        ):
            raise ValueError("edge relative-yaw pair must have unit norm")

    def _grounding(
        self,
        visual: torch.Tensor,
        mission: torch.Tensor,
    ) -> torch.Tensor:
        return self._grounding_with_logit(visual, mission)[0]

    def _grounding_with_logit(
        self,
        visual: torch.Tensor,
        mission: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_gate = self.grounding_target_gate(mission)
        raw = self.grounding_head(visual * target_gate)
        return self._bounded_grounding(raw), raw[:, 0]

    def _bounded_grounding(self, raw: torch.Tensor) -> torch.Tensor:
        visible = self.visible_activation(raw[:, :1])
        center = visible * self.center_activation(raw[:, 1:3])
        scale = visible * self.scale_activation(raw[:, 3:4])
        return torch.cat((visible, center, scale), dim=1)
