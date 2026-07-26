from __future__ import annotations

import torch

from flightrl.sixdof.puffer_observation import scale_previous_action_observation
from flightrl.sixdof.crash_selection import crash_replay_selection_metrics


class TailPolicy(torch.nn.Module):
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[:, -4:]


def test_previous_action_observation_scale_masks_tail_channels() -> None:
    observations = torch.ones(2, 28)

    scaled = scale_previous_action_observation(observations, 0.25)

    assert torch.allclose(scaled[:, :-4], torch.ones(2, 24))
    assert torch.allclose(scaled[:, -4:], torch.full((2, 4), 0.25))
    assert torch.allclose(observations[:, -4:], torch.ones(2, 4))


def test_crash_selection_uses_previous_action_observation_scale() -> None:
    replay = {"observations": torch.ones(8, 28), "target_actions": torch.zeros(8, 4)}

    full = crash_replay_selection_metrics(TailPolicy(), replay, action_abs_limit=0.5)
    masked = crash_replay_selection_metrics(TailPolicy(), replay, action_abs_limit=0.5, previous_action_observation_scale=0.0)

    assert full["crash_replay_action_abs_max"] == 1.0
    assert masked["crash_replay_action_abs_max"] == 0.0
