from __future__ import annotations

import torch
from torch import nn

from flightrl.mujoco.semantic_observation import SemanticStudentObservationLayout
from flightrl.navigation.spatial_memory import MAP_CHANNELS, SpatialMemoryConfig
from flightrl.vision import VisionObservationConfig


class SemanticVisionEncoder(nn.Module):
    def __init__(
        self,
        observation_size: int,
        hidden_size: int = 128,
        *,
        vision_config: VisionObservationConfig | None = None,
        memory_config: SpatialMemoryConfig | None = None,
    ) -> None:
        super().__init__()
        vision = vision_config or VisionObservationConfig(
            width=64,
            height=48,
            color_mode="grayscale",
            include_delta=True,
            include_motion_mask=True,
        )
        memory = memory_config or SpatialMemoryConfig()
        self.layout = SemanticStudentObservationLayout(vision, memory)
        if observation_size != self.layout.flat_dim:
            raise ValueError(
                f"semantic encoder expected {self.layout.flat_dim} observations, "
                f"got {observation_size}"
            )
        self.vision_shape = vision.shape
        self.map_shape = memory.shape
        self.vision_feature_dim = 16 * 3 * 4
        self.vision = nn.Sequential(
            nn.Conv2d(vision.channels, 8, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(12, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(len(MAP_CHANNELS), 8, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        state_size = self.layout.flat_dim - self.layout.proprioception_slice.start
        self.state = nn.Sequential(nn.Linear(state_size, 32), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(16 * 3 * 4 + 12 * 2 * 2 + 32 + 4, hidden_size),
            nn.GELU(),
        )
        coordinates = torch.linspace(1.0, -1.0, memory.local_size)
        forward, left = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("target_forward", forward.reshape(1, -1))
        self.register_buffer("target_left", left.reshape(1, -1))
        self.target_channel = MAP_CHANNELS.index("target_evidence")

    def _parts(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = observations.reshape(observations.shape[0], -1).float()
        images = values[:, self.layout.vision_slice].reshape(-1, *self.vision_shape)
        maps = values[:, self.layout.map_slice].reshape(-1, *self.map_shape)
        state = values[:, self.layout.proprioception_slice.start :]
        target = maps[:, self.target_channel].flatten(1).clamp_min(0.0)
        target_mass = target.sum(dim=1, keepdim=True)
        target_features = torch.cat(
            (
                target_mass.clamp(0.0, 1.0),
                target.max(dim=1, keepdim=True).values,
                (target * self.target_forward).sum(dim=1, keepdim=True)
                / target_mass.clamp_min(1e-6),
                (target * self.target_left).sum(dim=1, keepdim=True)
                / target_mass.clamp_min(1e-6),
            ),
            dim=1,
        )
        return (
            self.vision(images),
            self.spatial(maps),
            self.state(state),
            target_features,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        fused, _vision = self.forward_with_vision(observations)
        return fused

    def forward_with_vision(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vision, spatial, state, target = self._parts(observations)
        fused = self.fusion(torch.cat((vision, spatial, state, target), dim=1))
        return fused, vision

    def vision_features(self, observations: torch.Tensor) -> torch.Tensor:
        vision, _spatial, _state, _target = self._parts(observations)
        return vision

    def target_acquired(self, observations: torch.Tensor) -> torch.Tensor:
        acquired, _bearing = self.target_memory_direction(observations)
        return acquired

    def target_memory_direction(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = observations.reshape(observations.shape[0], -1).float()
        maps = values[:, self.layout.map_slice].reshape(-1, *self.map_shape)
        evidence = maps[:, self.target_channel].flatten(1).clamp_min(0.0)
        mass = evidence.sum(dim=1, keepdim=True)
        forward = (evidence * self.target_forward).sum(
            dim=1,
            keepdim=True,
        ) / mass.clamp_min(1e-6)
        left = (evidence * self.target_left).sum(
            dim=1,
            keepdim=True,
        ) / mass.clamp_min(1e-6)
        acquired = (mass > 0.0).to(values.dtype)
        return acquired, torch.atan2(left, forward)
