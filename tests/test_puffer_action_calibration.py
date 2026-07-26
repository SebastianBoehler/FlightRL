from __future__ import annotations

import torch

from flightrl.sixdof.action_calibration import scaled_action_head_state_dict


def test_scaled_action_head_state_dict_only_scales_decoder_mean() -> None:
    state = {
        "encoder.encoder.weight": torch.ones(2, 2),
        "decoder.decoder_mean.weight": torch.ones(2, 2),
        "decoder.decoder_mean.bias": torch.ones(2),
        "decoder.value_function.weight": torch.ones(1, 2),
    }

    scaled = scaled_action_head_state_dict(state, 0.5)

    assert torch.equal(scaled["encoder.encoder.weight"], torch.ones(2, 2))
    assert torch.equal(scaled["decoder.value_function.weight"], torch.ones(1, 2))
    assert torch.equal(scaled["decoder.decoder_mean.weight"], torch.full((2, 2), 0.5))
    assert torch.equal(scaled["decoder.decoder_mean.bias"], torch.full((2,), 0.5))
