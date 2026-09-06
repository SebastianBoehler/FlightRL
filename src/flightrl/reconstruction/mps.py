"""Explicit Apple GPU compatibility for LingBot's cached rotary frequencies."""

import torch


def prepare_rotary_cache(model):
    """Cast complex128 caches before transfer; MPS has no float64 storage.

    Upstream applies the real and imaginary components after casting to the
    float32 query dtype. Casting these immutable caches first preserves that
    arithmetic; this does not change model weights or substitute a backend.
    """
    changed = 0
    for module in model.modules():
        if type(module).__name__ == "WanRotaryPosEmbed":
            if module.freqs.dtype != torch.complex128:
                raise ValueError("Unexpected upstream rotary cache dtype")
            module.freqs = module.freqs.to(torch.complex64)
            changed += 1
    if not changed:
        raise ValueError("No expected LingBot rotary cache found")
    return changed
