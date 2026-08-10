from __future__ import annotations

import torch
from torch import nn

from flightrl.puffer4_edge_policy import HardGatedRecurrentCell
from flightrl.puffer4_edge_schema import (
    EDGE_FRAME_PIXELS,
    EDGE_HEIGHT,
    EDGE_TELEMETRY_BOUNDS,
    EDGE_TELEMETRY_DIM,
    EDGE_WIDTH,
    TELEMETRY_SPECS,
)

from .contract import (
    COVERAGE_MAXIMUM_YAW_RATE_DEG_S,
    COVERAGE_OBSERVATION_DIM,
)


CONTROLLED_TELEMETRY_INDICES = (15, 18)
LEARNED_DELTA_LIMIT = 2.0


class CoverageExplorationActor(nn.Module):
    """Simulation-only camera recurrent actor with no target or map input."""

    observation_size = COVERAGE_OBSERVATION_DIM
    action_size = 4

    def __init__(self, hidden_size: int = 48) -> None:
        super().__init__()
        if type(hidden_size) is not int or not 32 <= hidden_size <= 64:
            raise ValueError("coverage actor hidden size must be an integer in [32, 64]")
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
        self.fusion = nn.Sequential(
            nn.Linear(visual_size + EDGE_TELEMETRY_DIM, hidden_size),
            nn.ReLU6(),
        )
        self.recurrent = HardGatedRecurrentCell(hidden_size)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 2),
            nn.Hardtanh(-LEARNED_DELTA_LIMIT, LEARNED_DELTA_LIMIT),
        )
        nn.init.zeros_(self.action_head[0].weight)
        nn.init.zeros_(self.action_head[0].bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_state(
        self, batch_size: int, *, device: torch.device | None = None
    ) -> torch.Tensor:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("coverage batch size must be a positive integer")
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(observation, state)
        frame = observation[:, :EDGE_FRAME_PIXELS].reshape(
            -1, 1, EDGE_HEIGHT, EDGE_WIDTH
        )
        telemetry = observation[:, EDGE_FRAME_PIXELS:]
        visual = self.visual(frame)
        encoded = self.fusion(torch.cat((visual, telemetry), dim=1))
        next_state = self.recurrent(encoded, state)
        delta = self.action_head(next_state)
        previous_vx = telemetry[:, CONTROLLED_TELEMETRY_INDICES[0]]
        previous_yaw = telemetry[:, CONTROLLED_TELEMETRY_INDICES[1]] * (
            TELEMETRY_SPECS[18][2] / COVERAGE_MAXIMUM_YAW_RATE_DEG_S
        )
        previous = torch.stack((previous_vx, previous_yaw), dim=1).clamp(-1.0, 1.0)
        controlled = (previous + delta).clamp(-1.0, 1.0)
        zero = torch.zeros_like(controlled[:, 0])
        action = torch.stack((controlled[:, 0], zero, zero, controlled[:, 1]), dim=1)
        if not torch.isfinite(action).all() or not torch.isfinite(next_state).all():
            raise RuntimeError("coverage actor produced nonfinite action or state")
        return action, next_state

    def _validate_inputs(
        self, observation: torch.Tensor, state: torch.Tensor
    ) -> None:
        if observation.ndim != 2 or observation.shape[1] != COVERAGE_OBSERVATION_DIM:
            raise ValueError(
                f"coverage observation must have shape [batch, {COVERAGE_OBSERVATION_DIM}]"
            )
        if state.shape != (observation.shape[0], self.hidden_size):
            raise ValueError("coverage recurrent state shape is incompatible")
        if observation.dtype != torch.float32 or state.dtype != torch.float32:
            raise ValueError("coverage observation and state must be float32")
        if observation.device != state.device:
            raise ValueError("coverage observation and state must share a device")
        if not torch.isfinite(observation).all() or not torch.isfinite(state).all():
            raise ValueError("coverage observation and state must be finite")
        if torch.any((state < 0.0) | (state > 6.0)):
            raise ValueError("coverage recurrent state violates the [0, 6] invariant")
        frame = observation[:, :EDGE_FRAME_PIXELS]
        levels = frame * 15.0
        if torch.any((frame < 0.0) | (frame > 1.0)) or not torch.allclose(
            levels, levels.round(), atol=1.0e-6, rtol=0.0
        ):
            raise ValueError("coverage frame must contain exact normalized gray4 levels")
        telemetry = observation[:, EDGE_FRAME_PIXELS:]
        for index, (low, high) in enumerate(EDGE_TELEMETRY_BOUNDS):
            if torch.any((telemetry[:, index] < low) | (telemetry[:, index] > high)):
                raise ValueError("coverage telemetry violates normalized bounds")
        self._require_unit_vector(telemetry[:, 6:9], "body-up")
        self._require_unit_vector(telemetry[:, 13:15], "relative-yaw")

    @staticmethod
    def _require_unit_vector(values: torch.Tensor, label: str) -> None:
        norm = torch.linalg.vector_norm(values, dim=1)
        if not torch.allclose(norm, torch.ones_like(norm), atol=1.0e-4, rtol=0.0):
            raise ValueError(f"coverage {label} vector must have unit norm")
