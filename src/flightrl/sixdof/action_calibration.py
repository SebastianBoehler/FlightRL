from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch


ACTION_HEAD_KEYS = ("decoder.decoder_mean.weight", "decoder.decoder_mean.bias")


def scaled_action_head_state_dict(state_dict: Mapping[str, torch.Tensor], scale: float) -> dict[str, torch.Tensor]:
    if scale <= 0.0:
        raise ValueError("action head scale must be positive")
    output = {key: value.detach().clone() for key, value in state_dict.items()}
    for key in ACTION_HEAD_KEYS:
        if key not in output:
            raise ValueError(f"checkpoint is missing Puffer action head key {key!r}")
        output[key] = output[key] * float(scale)
    return output


def write_scaled_action_head_checkpoint(input_path: str | Path, output_path: str | Path, scale: float) -> None:
    state_dict = torch.load(input_path, map_location="cpu")
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scaled_action_head_state_dict(state_dict, scale), output)
