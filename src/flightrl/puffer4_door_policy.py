from __future__ import annotations

import torch
from torch import nn

from flightrl.puffer4_door_observation import DOOR_PROPRIO_DIM
from flightrl.semantic.door_observability import DoorObservabilityNet

DOOR_WIDTH = 64
DOOR_HEIGHT = 48
DOOR_CHANNELS = 3
DOOR_PRIVILEGED_DIM = 6
DOOR_PIXELS = DOOR_WIDTH * DOOR_HEIGHT
DOOR_POLICY_OBS_DIM = DOOR_CHANNELS * DOOR_PIXELS + DOOR_PROPRIO_DIM
DOOR_OBS_DIM = DOOR_POLICY_OBS_DIM + DOOR_PRIVILEGED_DIM


class FlightRLDoorEncoder(nn.Module):
    """Camera-only fixed-door encoder; privileged teacher values are excluded."""

    def __init__(self, observation_size: int, hidden_size: int = 96) -> None:
        super().__init__()
        if observation_size != DOOR_OBS_DIM:
            raise ValueError(
                f"fixed-door observation must have {DOOR_OBS_DIM} values, "
                f"got {observation_size}"
            )
        self.visual = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(12 * 3 * 4 + DOOR_PROPRIO_DIM + 4, hidden_size),
            nn.GELU(),
        )
        self.grounder = DoorObservabilityNet()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encode_with_grounding(observations)
        return hidden

    def encode_with_grounding(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        deployable = observations.float()[:, :DOOR_POLICY_OBS_DIM]
        images = deployable[:, : DOOR_CHANNELS * DOOR_PIXELS].reshape(
            -1,
            DOOR_CHANNELS,
            DOOR_HEIGHT,
            DOOR_WIDTH,
        )
        proprio = deployable[:, DOOR_CHANNELS * DOOR_PIXELS :]
        visual = self.visual(images)
        grounding = self.grounder(images[:, :1])
        hidden = self.fusion(
            torch.cat((visual, proprio, torch.sigmoid(grounding)), dim=1)
        )
        return hidden, grounding

    def predict_grounding(self, observations: torch.Tensor) -> torch.Tensor:
        deployable = observations.float()[:, :DOOR_POLICY_OBS_DIM]
        images = deployable[:, : DOOR_CHANNELS * DOOR_PIXELS].reshape(
            -1,
            DOOR_CHANNELS,
            DOOR_HEIGHT,
            DOOR_WIDTH,
        )
        return self.grounder(images[:, :1])

    def load_observability_checkpoint(self, checkpoint: dict) -> None:
        contract = checkpoint.get("frame_contract", {})
        expected = {
            "width": DOOR_WIDTH,
            "height": DOOR_HEIGHT,
            "channels": 1,
            "quantization_levels": 16,
        }
        if any(contract.get(key) != value for key, value in expected.items()):
            raise ValueError("observability checkpoint frame contract does not match D1")
        source = checkpoint["state_dict"]
        self.grounder.load_state_dict(source)
        with torch.no_grad():
            self.visual[0].weight.zero_()
            self.visual[0].weight[:, :1].copy_(
                source["encoder.0.weight"][: self.visual[0].out_channels]
            )
            self.visual[0].bias.copy_(
                source["encoder.0.bias"][: self.visual[0].out_channels]
            )
