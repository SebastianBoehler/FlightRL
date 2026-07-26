from __future__ import annotations

from pathlib import Path

import torch

from flightrl.vision import (
    CompactVisionActionPolicy,
    VisionActionPolicyMetadata,
    load_vision_action_policy,
    save_vision_action_policy,
)


def test_compact_vision_action_checkpoint_round_trip(tmp_path: Path) -> None:
    metadata = VisionActionPolicyMetadata(channels=3, height=48, width=64, hidden_size=16)
    policy = CompactVisionActionPolicy(metadata)
    observations = torch.zeros(2, 3, 48, 64)
    checkpoint = tmp_path / "vision_action.pt"

    save_vision_action_policy(checkpoint, policy, training={"samples": 2})
    loaded = load_vision_action_policy(checkpoint)

    assert loaded(observations).shape == (2, 3)
    torch.testing.assert_close(loaded(observations), policy(observations))
