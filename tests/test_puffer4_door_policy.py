from __future__ import annotations

import torch

from flightrl.puffer4_door_policy import (
    DOOR_OBS_DIM,
    DOOR_POLICY_OBS_DIM,
    FlightRLDoorEncoder,
)
from flightrl.puffer4_door_imitation import freeze_door_grounder
from flightrl.semantic.door_observability import DoorObservabilityNet


def test_door_encoder_excludes_privileged_teacher_tail() -> None:
    torch.manual_seed(3)
    encoder = FlightRLDoorEncoder(DOOR_OBS_DIM, hidden_size=32)
    observations = torch.rand(2, DOOR_OBS_DIM)
    changed = observations.clone()
    changed[:, DOOR_POLICY_OBS_DIM:] = torch.tensor(
        (
            (1.0, -1.0, 1.0, 0.2, 0.3, 0.4),
            (-1.0, 1.0, 0.0, 0.8, 0.7, 0.6),
        )
    )

    assert torch.equal(encoder(observations), encoder(changed))


def test_door_encoder_loads_exact_observability_contract() -> None:
    source = DoorObservabilityNet()
    encoder = FlightRLDoorEncoder(DOOR_OBS_DIM, hidden_size=32)
    checkpoint = {
        "state_dict": source.state_dict(),
        "frame_contract": {
            "width": 64,
            "height": 48,
            "channels": 1,
            "quantization_levels": 16,
        },
    }

    encoder.load_observability_checkpoint(checkpoint)

    target = encoder.grounder.state_dict()
    assert all(
        torch.equal(target[key], value)
        for key, value in source.state_dict().items()
    )
    assert torch.equal(
        encoder.visual[0].weight[:, :1],
        source.encoder[0].weight[: encoder.visual[0].out_channels],
    )
    assert torch.count_nonzero(encoder.visual[0].weight[:, 1:]) == 0


def test_grounder_can_be_frozen_before_action_training() -> None:
    class Policy:
        encoder = FlightRLDoorEncoder(DOOR_OBS_DIM, hidden_size=32)

    freeze_door_grounder(Policy())

    assert not any(
        parameter.requires_grad
        for parameter in Policy.encoder.grounder.parameters()
    )
