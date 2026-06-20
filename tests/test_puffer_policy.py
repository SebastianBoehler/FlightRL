from __future__ import annotations

import torch

from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy, infer_metadata


def test_puffer_policy_infers_metadata_and_loads_state_dict() -> None:
    original = PufferSixDofPolicy(PufferPolicyMetadata(observation_dim=28, hidden_size=8, action_dim=4, num_layers=2))
    loaded = PufferSixDofPolicy.from_state_dict(original.state_dict())

    assert infer_metadata(original.state_dict()).observation_dim == 28
    assert loaded.metadata.hidden_size == 8
    assert loaded.metadata.action_dim == 4


def test_puffer_policy_outputs_clamped_actions() -> None:
    policy = PufferSixDofPolicy(PufferPolicyMetadata(observation_dim=3, hidden_size=4, action_dim=2, num_layers=1))
    with torch.no_grad():
        policy.decoder.decoder_mean.bias[:] = torch.tensor([3.0, -3.0])

    actions = policy(torch.zeros(5, 3))

    assert actions.shape == (5, 2)
    assert torch.max(actions) <= 1.0
    assert torch.min(actions) >= -1.0
