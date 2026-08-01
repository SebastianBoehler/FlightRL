from __future__ import annotations

import torch

from flightrl.puffer4_door_policy import (
    DOOR_HEIGHT,
    DOOR_PIXELS,
    DOOR_POLICY_OBS_DIM,
    DOOR_WIDTH,
)


def collect_grounding_replay(
    streams: list[tuple],
    *,
    batches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    frames = []
    labels = []
    for _ in range(batches):
        for vec, observations in streams:
            vec.reset()
            current = observations[:, :DOOR_PIXELS].reshape(
                -1,
                1,
                DOOR_HEIGHT,
                DOOR_WIDTH,
            )
            frames.append(torch.round(current * 255.0).to(torch.uint8))
            labels.append(
                observations[:, DOOR_POLICY_OBS_DIM + 2 :].clone()
            )
    return torch.cat(frames), torch.cat(labels)
