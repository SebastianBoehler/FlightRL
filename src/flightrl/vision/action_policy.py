from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class VisionActionPolicyMetadata:
    channels: int
    height: int
    width: int
    hidden_size: int = 64
    velocity_scale_m_s: float = 0.15
    yawrate_scale_deg_s: float = 60.0
    contract_json: str = ""


class CompactVisionActionPolicy(nn.Module):
    def __init__(self, metadata: VisionActionPolicyMetadata) -> None:
        super().__init__()
        self.metadata = metadata
        self.encoder = nn.Sequential(
            nn.Conv2d(metadata.channels, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4)),
        )
        self.action_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 3 * 4, metadata.hidden_size),
            nn.ReLU(),
            nn.Linear(metadata.hidden_size, 3),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        expected = (self.metadata.channels, self.metadata.height, self.metadata.width)
        if tuple(observations.shape[1:]) != expected:
            raise ValueError(f"expected BCHW observations with trailing shape {expected}, got {tuple(observations.shape)}")
        return self.action_head(self.encoder(observations.float()))

    def physical_actions(self, observations: torch.Tensor) -> torch.Tensor:
        scale = observations.new_tensor(
            [
                self.metadata.velocity_scale_m_s,
                self.metadata.velocity_scale_m_s,
                self.metadata.yawrate_scale_deg_s,
            ]
        )
        return self(observations) * scale


def save_vision_action_policy(
    path: str | Path,
    policy: CompactVisionActionPolicy,
    *,
    training: Mapping[str, object],
) -> None:
    torch.save(
        {
            "format": "flightrl.vision_action.v1",
            "metadata": asdict(policy.metadata),
            "state_dict": policy.state_dict(),
            "training": dict(training),
        },
        path,
    )


def load_vision_action_policy(path: str | Path) -> CompactVisionActionPolicy:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "flightrl.vision_action.v1":
        raise ValueError("unsupported vision-action checkpoint format")
    policy = CompactVisionActionPolicy(VisionActionPolicyMetadata(**payload["metadata"]))
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy
